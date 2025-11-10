import express from "express";
import http from "http";
import { WebSocketServer } from "ws";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const server = http.createServer(app);

app.use(express.static(path.join(__dirname, "public")));

const wss = new WebSocketServer({ server });

wss.on("connection", (ws) => {
  console.log("Client connected");

  ws.on("close", () => console.log("Client disconnected"));

  ws.on("message", (message) => {
    console.log("Received from worker:", message.toString());

    wss.clients.forEach((client) => {
      if (client.readyState === ws.OPEN) client.send(message.toString());
    });
  });
});

const PORT = 9000;
server.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
});
