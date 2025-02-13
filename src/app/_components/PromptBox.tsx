"use client";

import { useState } from "react";
import type { Route } from "~/app/api/chat/route";
import { api } from "~/trpc/react";

export function PromptBox() {
  const [prompt, setPrompt] = useState("");
  const [isPending, setIsPending] = useState(false);
  const [result, setResult] = useState<Route[] | null>(null);

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
      setResult(data as Route[]);
    } finally {
      setIsPending(false);
    }
  };

  return (
    <div className="mx-auto max-w-4xl p-6">
      <form onSubmit={handleSubmit} className="mb-8 flex flex-col gap-2">
        <input
          type="text"
          placeholder="Enter your prompt..."
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          className="w-full rounded-lg border border-gray-700 bg-gray-800 px-4 py-3 text-white placeholder-gray-400 focus:border-transparent focus:ring-2 focus:ring-blue-500"
        />
        <button
          type="submit"
          className="rounded-lg bg-blue-600 px-10 py-3 font-semibold text-white transition hover:bg-blue-700 disabled:opacity-50"
          disabled={isPending}
        >
          {isPending ? "Processing..." : "Submit"}
        </button>
      </form>

      {result && result.length > 0 && (
        <div className="overflow-x-auto rounded-lg border border-gray-700">
          <table className="min-w-full divide-y divide-gray-700">
            <thead className="bg-gray-800">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-300">
                  Name
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-300">
                  Quality
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-300">
                  Type
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-300">
                  Grade
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-300">
                  Area Name
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-700 bg-gray-900">
              {result.map((item: Route, index: number) => (
                <tr key={index} className="hover:bg-gray-800">
                  <td className="whitespace-nowrap px-6 py-4 text-gray-300">
                    {item.name}
                  </td>
                  <td className="whitespace-nowrap px-6 py-4 text-gray-300">
                    {item.quality}
                  </td>
                  <td className="whitespace-nowrap px-6 py-4 text-gray-300">
                    {item.type}
                  </td>
                  <td className="whitespace-nowrap px-6 py-4 text-gray-300">
                    {item.grade}
                  </td>
                  <td className="whitespace-nowrap px-6 py-4 text-gray-300">
                    {item.area_name}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
