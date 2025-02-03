import { api, HydrateClient } from "~/trpc/server";
import { PromptBox } from "../_components/PromptBox";

export default async function Home() {
  const hello = await api.post.hello({ text: "from tRPC" });

  return (
    <HydrateClient>
      <div className="container flex flex-col items-center justify-center gap-12 bg-black px-4 py-16 text-white">
        <h1 className="my-10 text-5xl font-extrabold tracking-tight sm:text-[5rem]">
          {hello.greeting}
        </h1>
        <PromptBox />
      </div>
    </HydrateClient>
  );
}
