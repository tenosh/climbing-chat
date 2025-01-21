import { z } from "zod";
import { createTRPCRouter, protectedProcedure } from "~/server/api/trpc";
import { openai } from "@ai-sdk/openai";
import { streamText } from "ai";

export const chatRouter = createTRPCRouter({
  sendMessage: protectedProcedure
    .input(
      z.object({
        messages: z.array(
          z.object({
            content: z.string(),
            role: z.enum(["user", "assistant"]),
          }),
        ),
      }),
    )
    .mutation(async ({ input }) => {
      const result = await streamText({
        model: openai("gpt-4"),
        messages: input.messages,
      });

      return result.toDataStreamResponse();
    }),
});
