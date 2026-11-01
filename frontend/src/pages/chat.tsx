import { useCallback, useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useNavigate, useParams } from "@tanstack/react-router";
import { api } from "@/api/client";
import { ChatWindow } from "@/features/chat/chat-window";
import { useChat } from "@/hooks/useChat";
import { useChatStore } from "@/store/chat";
import type { Message } from "@/types";

export function ChatPage() {
  const navigate = useNavigate();
  const { conversationId: routeConversationId } = useParams({ strict: false });
  const { startConversation } = useChat();
  const {
    activeConversationId,
    messages,
    isStreaming,
    streamingContent,
    appendMessage,
    appendStreamContent,
    setStreaming,
    clearStreamContent,
    setActiveConversation,
    addMessages,
  } = useChatStore();
  const [streamError, setStreamError] = useState<string | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  const messagesQuery = useQuery({
    queryKey: ["messages", activeConversationId],
    queryFn: () =>
      api.get<Message[]>(
        `/chats/conversations/${activeConversationId}/messages`
      ),
    enabled: Boolean(activeConversationId),
  });

  useEffect(() => {
    setActiveConversation(routeConversationId || null);
  }, [routeConversationId, setActiveConversation]);

  useEffect(() => {
    if (activeConversationId && messagesQuery.data) {
      addMessages(activeConversationId, messagesQuery.data);
    }
  }, [activeConversationId, addMessages, messagesQuery.data]);

  useEffect(
    () => () => abortControllerRef.current?.abort(),
    []
  );

  const handleSend = useCallback(
    async (content: string) => {
      if (isStreaming) return;
      setStreamError(null);

      let conversationId = activeConversationId;
      if (!conversationId) {
        try {
          const conversation = await startConversation(content.slice(0, 60));
          conversationId = conversation.id;
          navigate({
            to: "/chat/$conversationId",
            params: { conversationId },
          });
        } catch (error) {
          setStreamError(
            error instanceof Error
              ? error.message
              : "Unable to start a new conversation."
          );
          return;
        }
      }

      appendMessage(conversationId, {
        id: `temp-${Date.now()}`,
        conversation_id: conversationId,
        role: "user",
        content,
        model_id: null,
        provider_id: null,
        tool_calls: null,
        tool_call_id: null,
        input_tokens: null,
        output_tokens: null,
        total_tokens: null,
        latency_ms: null,
        finish_reason: null,
        created_at: new Date().toISOString(),
        parent_id: null,
      });

      setStreaming(true);
      clearStreamContent();
      const controller = new AbortController();
      abortControllerRef.current = controller;
      try {
        const response = await api.stream(
          `/chats/conversations/${conversationId}/stream`,
          { content },
          controller.signal
        );
        const reader = response.body!.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const frames = buffer.split(/\r?\n\r?\n/);
          buffer = frames.pop() || "";

          for (const frame of frames) {
            let eventName = "message";
            const dataLines: string[] = [];
            for (const line of frame.split(/\r?\n/)) {
              if (line.startsWith("event:")) {
                eventName = line.slice(6).trim();
              } else if (line.startsWith("data:")) {
                dataLines.push(line.slice(5).trimStart());
              }
            }

            if (eventName === "token" && dataLines.length) {
              const chunk = JSON.parse(dataLines.join("\n"));
              appendStreamContent(chunk.content || "");
            } else if (eventName === "message" && dataLines.length) {
              appendMessage(conversationId, JSON.parse(dataLines.join("\n")));
            } else if (eventName === "error" && dataLines.length) {
              const data = JSON.parse(dataLines.join("\n"));
              throw new Error(data.detail || "The assistant could not respond.");
            }
          }
        }
      } catch (error) {
        if (!controller.signal.aborted) {
          setStreamError(
            error instanceof Error
              ? error.message
              : "The assistant could not respond."
          );
        }
      } finally {
        abortControllerRef.current = null;
        setStreaming(false);
        clearStreamContent();
      }
    },
    [
      activeConversationId,
      appendMessage,
      appendStreamContent,
      clearStreamContent,
      isStreaming,
      navigate,
      setStreaming,
      startConversation,
    ]
  );

  const handleStop = useCallback(() => {
    abortControllerRef.current?.abort();
  }, []);

  return (
    <ChatWindow
      messages={activeConversationId ? messages[activeConversationId] || [] : []}
      isStreaming={isStreaming}
      streamingContent={streamingContent}
      error={
        streamError ||
        (messagesQuery.error instanceof Error
          ? messagesQuery.error.message
          : null)
      }
      onSend={handleSend}
      onStop={handleStop}
      activeConversationId={activeConversationId}
    />
  );
}
