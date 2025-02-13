import { api, HydrateClient } from "~/trpc/server";

export default async function Home() {
  const hello = await api.post.hello({ text: "from tRPC" });

  return (
    <HydrateClient>
      <div className="container mx-auto py-8">{/* <PromptBox /> */}</div>
    </HydrateClient>
  );
}
