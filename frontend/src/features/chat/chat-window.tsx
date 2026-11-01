import { useEffect, useRef } from "react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { MessageBubble } from "./message-bubble";
import { MessageInput } from "./message-input";
import { MessageSquare } from "lucide-react";
import type { Message } from "@/types";

interface ChatWindowProps {
  messages: Message[];
  isStreaming: boolean;
  streamingContent: string;
  error?: string | null;
  onSend: (content: string) => void;
  onStop: () => void;
  activeConversationId: string | null;
}

export function ChatWindow({
  messages,
  isStreaming,
  streamingContent,
  error,
  onSend,
  onStop,
  activeConversationId,
}: ChatWindowProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamingContent]);

  if (!activeConversationId) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center">
          <MessageSquare className="mx-auto h-12 w-12 text-muted-foreground/50" />
          <h2 className="mt-4 text-xl font-semibold">Select a conversation</h2>
          <p className="mt-2 text-sm text-muted-foreground">
            Choose a conversation from the sidebar or start a new one
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      <ScrollArea className="flex-1 px-4" ref={scrollRef}>
        <div className="mx-auto max-w-4xl py-4">
          {messages.length === 0 && (
            <div className="flex items-center justify-center h-full min-h-[400px]">
              <div className="text-center">
                <MessageSquare className="mx-auto h-10 w-10 text-muted-foreground/40" />
                <p className="mt-3 text-sm text-muted-foreground">
                  Send a message to start the conversation
                </p>
              </div>
            </div>
          )}

          {messages.map((message) => (
            <MessageBubble key={message.id} message={message} />
          ))}

          {isStreaming && (
            <MessageBubble
              message={{
                id: "streaming",
                conversation_id: activeConversationId,
                role: "assistant",
                content: "",
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
              }}
              isStreaming
              streamingContent={streamingContent}
            />
          )}

          {error && (
            <p role="alert" className="py-2 text-sm text-destructive">
              {error}
            </p>
          )}

          <div ref={bottomRef} />
        </div>
      </ScrollArea>

      <MessageInput
        onSend={onSend}
        onStop={onStop}
        isStreaming={isStreaming}
        disabled={isStreaming}
      />
    </div>
  );
}
