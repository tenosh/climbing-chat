"use client";

import { useState } from "react";

import { api } from "~/trpc/react";

export function PromptBox() {
  const [prompt, setPrompt] = useState("");
  const [isPending, setIsPending] = useState(false);

  const generateQuery = api.chat.generateQuery.useMutation();
  const runQuery = api.chat.runQuery.useMutation();
  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setIsPending(true);
    try {
      const query = await generateQuery.mutateAsync({ prompt });
      if (query === undefined) {
        setIsPending(false);
        return;
      }
      const data = await runQuery.mutateAsync({ query });
      console.log(data);
    } finally {
      setIsPending(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-2">
      <input
        type="text"
        placeholder="prompt"
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        className="w-full rounded-full px-4 py-2 text-black"
      />
      <button
        type="submit"
        className="rounded-full bg-white/10 px-10 py-3 font-semibold transition hover:bg-white/20"
        disabled={isPending}
      >
        {isPending ? "Submitting..." : "Submit"}
      </button>
    </form>
  );
}
