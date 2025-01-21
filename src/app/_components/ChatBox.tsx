"use client";

import { useChat } from "ai/react";

export function ChatBox() {
  const { messages, input, handleInputChange, handleSubmit, isLoading } =
    useChat();

  return (
    <div className="mx-auto flex w-full max-w-2xl flex-col rounded-lg bg-white/5">
      <div className="flex h-[500px] flex-col gap-4 overflow-y-auto p-4">
        {messages.map((m) => (
          <div
            key={m.id}
            className={`whitespace-pre-wrap rounded-lg p-4 ${
              m.role === "user"
                ? "ml-auto max-w-[80%] bg-blue-500/50"
                : "mr-auto max-w-[80%] bg-gray-600/50"
            }`}
          >
            {m.content}
          </div>
        ))}
      </div>

      <form onSubmit={handleSubmit} className="border-t border-white/10 p-4">
        <div className="relative">
          <input
            className="w-full rounded-lg bg-white/10 p-3 text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
            value={input}
            placeholder="Type your message..."
            onChange={handleInputChange}
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={isLoading}
            className="absolute right-3 top-1/2 -translate-y-1/2 rounded-md bg-purple-500 px-4 py-1 transition-colors hover:bg-purple-600 disabled:opacity-50"
          >
            {isLoading ? "Thinking..." : "Send"}
          </button>
        </div>
      </form>
    </div>
  );
}
