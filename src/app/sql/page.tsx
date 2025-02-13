import { HydrateClient } from "~/trpc/server";

export default async function Home() {
  return (
    <HydrateClient>
      <div className="container mx-auto py-8">{/* <PromptBox /> */}</div>
    </HydrateClient>
  );
}
