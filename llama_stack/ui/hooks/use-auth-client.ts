import { useSession } from "next-auth/react";
import { useMemo } from "react";
import LlamaStackClient from "llama-stack-client";

export function useAuthClient() {
  const { data: session } = useSession();

  const client = useMemo(() => {
    const clientHostname =
      typeof window !== "undefined" ? window.location.origin : "";

    const options: any = {
      baseURL: `${clientHostname}/api`,
      defaultHeaders: {
        "X-Telemetry-Service": "llama-stack-ui",
        "X-Telemetry-Version": "1.0.0",
      },
    };

    if (session?.accessToken) {
      options.apiKey = session.accessToken;
    }

    return new LlamaStackClient(options);
  }, [session?.accessToken]);

  return client;
}
