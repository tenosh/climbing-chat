import { api, HydrateClient } from "~/trpc/server";
import { PromptBox } from "../_components/PromptBox";

export default async function Home() {
  const hello = await api.post.hello({ text: "from tRPC" });

  return (
    <HydrateClient>
      <div className="container mx-auto py-8">
        <PromptBox />
      </div>
    </HydrateClient>
  );
}
