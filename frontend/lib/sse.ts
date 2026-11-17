// A tiny Server-Sent Events parser.
//
// The browser's EventSource only supports GET, but /chat/stream is a POST,
// so we read the response body ourselves. Network chunks can end anywhere —
// mid-line or mid-event — so the parser keeps a buffer and only returns
// events once their terminating blank line ("\n\n") has arrived.

export type SSEEvent = { event: string; data: string };

export function createSSEParser() {
  let buffer = "";

  return function push(chunk: string): SSEEvent[] {
    buffer = (buffer + chunk).replace(/\r\n/g, "\n");
    const events: SSEEvent[] = [];

    let end: number;
    while ((end = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);

      let event = "message";
      const data: string[] = [];
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
        // lines starting with ":" are comments (keep-alives); ignore them
      }
      if (data.length > 0) events.push({ event, data: data.join("\n") });
    }
    return events;
  };
}
