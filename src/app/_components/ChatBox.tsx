"use client";

import { useChat } from "@ai-sdk/react";
import ReactMarkdown from "react-markdown";

export function ChatBox() {
  const { messages, input, handleInputChange, handleSubmit, isLoading } =
    useChat({
      maxSteps: 10,
      initialMessages: [
        {
          id: "welcome",
          role: "assistant",
          content: `¡Qué onda! Soy Cactux, su guía más mamón de Guadalcázar. 🌵

A ver, estos son los pedos con los que te puedo ayudar (échale ganas):

🧗‍♂️ Beta de Rutas
- "¿Cuáles son las rutas más chidas de Las Candelas?"
- "Cuéntame de las rutas de grado 12 en San Caye"
- "¿Cuál es la ruta más dura en Panales?"

📍 Sectores de Escalada
- "¿Qué sectores hay en Guadalcázar?"
- "¿Cuál es el pronostico del clima en Salitre?"
- "Explícame el pedo de San Cayetano"
- "¿Cómo está el approach a Zelda?"

🏨 Beta Local
- "¿Dónde me puedo quedar a dormir en Guadalcázar?"
- "¿Dónde hay cheve y tacos chidos por aquí?"

Va, pregúntame lo que quieras...`,
        },
      ],
    });

  return (
    <div className="mx-auto flex w-full max-w-2xl flex-col rounded-lg bg-white/5">
      <div className="flex h-[500px] flex-col gap-4 overflow-y-auto p-4">
        {messages.map((m) => (
          <div
            key={m.id}
            className={`my-2 ${
              m.role === "user" ? "ml-auto" : "mr-auto"
            } max-w-[80%]`}
          >
            <div
              className={`rounded-2xl p-3 shadow-sm ${
                m.role === "user"
                  ? "rounded-br-sm bg-blue-500 text-white"
                  : "rounded-bl-sm bg-white text-black"
              }`}
            >
              <div className="font-bold">
                {m.role === "user" ? "escalador" : "cactux"}
              </div>
              {m.parts ? (
                <div>
                  {m.parts.map((part, index) => {
                    if (part.type === "text") {
                      return (
                        <ReactMarkdown
                          key={index}
                          className={
                            m.role === "user" ? "text-white" : "text-black"
                          }
                        >
                          {part.text}
                        </ReactMarkdown>
                      );
                    }
                    if (part.type === "tool-invocation") {
                      return (
                        <div key={index}>
                          tool name: {part.toolInvocation.toolName}
                        </div>
                      );
                    }
                    return null;
                  })}
                </div>
              ) : (
                <div className="whitespace-pre-wrap">{m.content}</div>
              )}
            </div>
          </div>
        ))}
      </div>

      <form onSubmit={handleSubmit} className="border-t border-white/10 p-4">
        <div className="relative">
          <input
            className="w-full rounded-lg bg-white/10 p-3 text-white focus:outline-none focus:ring-2 focus:ring-purple-500"
            value={input}
            placeholder="Escribe tu mensaje..."
            onChange={handleInputChange}
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={isLoading}
            className="absolute right-3 top-1/2 -translate-y-1/2 rounded-md bg-purple-500 px-4 py-1 transition-colors hover:bg-purple-600 disabled:opacity-50"
          >
            {isLoading ? "Pensando..." : "Enviar"}
          </button>
        </div>
      </form>
    </div>
  );
}
