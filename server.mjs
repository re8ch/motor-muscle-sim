import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL(".", import.meta.url));
const port = Number(process.env.PORT || 4173);

const types = new Map([
  [".html", "text/html; charset=utf-8"],
  [".css", "text/css; charset=utf-8"],
  [".js", "text/javascript; charset=utf-8"],
  [".json", "application/json; charset=utf-8"],
  [".svg", "image/svg+xml; charset=utf-8"]
]);

const server = createServer(async (req, res) => {
  try {
    const url = new URL(req.url || "/", `http://${req.headers.host}`);
    const requested = url.pathname === "/" ? "/index.html" : decodeURIComponent(url.pathname);
    const safePath = normalize(requested).replace(/^(\.\.[/\\])+/, "");
    const filePath = join(root, safePath);
    if (!filePath.startsWith(root)) {
      res.writeHead(403);
      res.end("Forbidden");
      return;
    }
    const body = await readFile(filePath);
    res.writeHead(200, {
      "content-type": types.get(extname(filePath)) || "application/octet-stream",
      "cache-control": "no-store"
    });
    res.end(body);
  } catch (error) {
    res.writeHead(404, { "content-type": "text/plain; charset=utf-8" });
    res.end("Not found");
  }
});

server.listen(port, () => {
  console.log(`motor-muscle-sim listening on http://127.0.0.1:${port}`);
});
