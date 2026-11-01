import { useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/api/client";
import { useChatStore } from "@/store/chat";
import type {
  Chat,
  Conversation,
  Message,
  PaginatedResponse,
} from "@/types";

export function useChat() {
  const queryClient = useQueryClient();
  const {
    conversations,
    activeConversationId,
    messages,
    isStreaming,
    streamingContent,
    setActiveConversation,
    setConversations,
    appendMessage,
    setStreaming,
    appendStreamContent,
    clearStreamContent,
    addConversation,
  } = useChatStore();

  const conversationsQuery = useQuery({
    queryKey: ["conversations"],
    queryFn: () =>
      api.get<PaginatedResponse<Conversation>>("/chats/conversations"),
    refetchOnMount: true,
  });

  useEffect(() => {
    if (conversationsQuery.data) {
      setConversations(conversationsQuery.data.items);
    }
  }, [conversationsQuery.data, setConversations]);

  const startConversation = async (title: string) => {
    const chat = await api.post<Chat>("/chats", { title });
    const conversation = await api.post<Conversation>(
      "/chats/conversations",
      { chat_id: chat.id, title }
    );
    addConversation(conversation);
    setActiveConversation(conversation.id);
    queryClient.invalidateQueries({ queryKey: ["conversations"] });
    return conversation;
  };

  const createConversation = useMutation({
    mutationFn: (data: { chat_id: string; title: string }) =>
      api.post<Conversation>("/chats/conversations", data),
    onSuccess: (newConversation) => {
      addConversation(newConversation);
      setActiveConversation(newConversation.id);
      queryClient.invalidateQueries({ queryKey: ["conversations"] });
    },
  });

  const sendMessage = useMutation({
    mutationFn: async (data: {
      conversationId: string;
      content: string;
    }) => {
      return api.post<Message>(
        `/chats/conversations/${data.conversationId}/messages`,
        { content: data.content }
      );
    },
    onSuccess: (message) => {
      if (activeConversationId) {
        appendMessage(activeConversationId, message);
      }
    },
  });

  return {
    conversations,
    activeConversationId,
    messages,
    isStreaming,
    streamingContent,
    conversationsQuery,
    setActiveConversation,
    startConversation,
    createConversation: createConversation.mutateAsync,
    sendMessage: sendMessage.mutateAsync,
    clearStreamContent,
    appendStreamContent,
    setStreaming,
    appendMessage,
  };
}
