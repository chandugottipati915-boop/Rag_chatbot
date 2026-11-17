// A server component: it renders once on the server and hands off to the
// interactive client component, which owns all the chat state.
import ClientChat from "@/components/ClientChat";

export default function Home() {
  return <ClientChat />;
}
