const state = {
  userId: getOrCreateNumber("agent_user_id"),
  sessionId: getOrCreateNumber("agent_session_id"),
  busy: false,
};

const nodes = {
  messages: document.querySelector("#messages"),
  composer: document.querySelector("#composer"),
  input: document.querySelector("#messageInput"),
  sendButton: document.querySelector("#sendButton"),
  newSessionButton: document.querySelector("#newSessionButton"),
  sessionLabel: document.querySelector("#sessionLabel"),
  userIdLabel: document.querySelector("#userIdLabel"),
  sessionIdLabel: document.querySelector("#sessionIdLabel"),
  requestState: document.querySelector("#requestState"),
  healthBadge: document.querySelector("#healthBadge"),
  traceText: document.querySelector("#traceText"),
};

function getOrCreateNumber(key) {
  const existing = Number(window.localStorage.getItem(key));
  if (Number.isInteger(existing) && existing > 0) {
    return existing;
  }
  const value = Math.floor(Date.now() + Math.random() * 100000);
  window.localStorage.setItem(key, String(value));
  return value;
}

function setSession(sessionId) {
  state.sessionId = sessionId;
  window.localStorage.setItem("agent_session_id", String(sessionId));
  nodes.sessionLabel.textContent = `session:${sessionId}`;
  nodes.userIdLabel.textContent = String(state.userId);
  nodes.sessionIdLabel.textContent = String(sessionId);
}

function setBusy(value, label = value ? "processing" : "idle") {
  state.busy = value;
  nodes.sendButton.disabled = value;
  nodes.input.disabled = value;
  nodes.requestState.textContent = label;
}

function renderMessages(messages) {
  nodes.messages.innerHTML = "";
  if (!messages.length) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "开始对话";
    nodes.messages.append(empty);
    return;
  }
  for (const message of messages) {
    appendMessage(message.role, message.content);
  }
}

function appendMessage(role, content) {
  nodes.messages.querySelector(".empty")?.remove();
  const item = document.createElement("article");
  item.className = `message ${role === "user" ? "user" : "assistant"}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = content || "";

  item.append(bubble);
  nodes.messages.append(item);
  nodes.messages.scrollTop = nodes.messages.scrollHeight;
}

async function checkHealth() {
  try {
    const response = await fetch("/api/health");
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    nodes.healthBadge.textContent = "ok";
    nodes.healthBadge.className = "ok";
  } catch {
    nodes.healthBadge.textContent = "error";
    nodes.healthBadge.className = "error";
  }
}

async function loadHistory() {
  const response = await fetch(
    `/api/sessions/${state.sessionId}/messages?user_id=${state.userId}`,
  );
  if (!response.ok) {
    renderMessages([]);
    return;
  }
  const data = await response.json();
  renderMessages(Array.isArray(data.messages) ? data.messages : []);
}

async function sendMessage(content) {
  appendMessage("user", content);
  setBusy(true, "queued");
  nodes.traceText.textContent = "request accepted";
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_id: state.userId,
        session_id: state.sessionId,
        message: content,
      }),
    });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const data = await response.json();
    await waitForTurn(data.turn_id);
  } catch (error) {
    appendMessage("assistant", "请求失败，请稍后重试。");
    nodes.traceText.textContent = error instanceof Error ? error.message : "failed";
  } finally {
    setBusy(false);
    nodes.input.focus();
  }
}

async function waitForTurn(turnId) {
  if (!turnId || turnId === "inline") {
    return;
  }
  for (;;) {
    const response = await fetch(`/api/turns/${turnId}`);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const data = await response.json();
    setBusy(true, data.status || "processing");
    nodes.traceText.textContent = data.status || "processing";
    if (data.status === "done") {
      appendMessage("assistant", data.answer || "");
      nodes.traceText.textContent = "done";
      return;
    }
    if (data.status === "failed") {
      throw new Error(data.error || "turn failed");
    }
    await new Promise((resolve) => window.setTimeout(resolve, 750));
  }
}

nodes.composer.addEventListener("submit", (event) => {
  event.preventDefault();
  const content = nodes.input.value.trim();
  if (!content || state.busy) {
    return;
  }
  nodes.input.value = "";
  sendMessage(content);
});

nodes.input.addEventListener("input", () => {
  nodes.input.style.height = "auto";
  nodes.input.style.height = `${Math.min(nodes.input.scrollHeight, 160)}px`;
});

nodes.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    nodes.composer.requestSubmit();
  }
});

nodes.newSessionButton.addEventListener("click", () => {
  setSession(Math.floor(Date.now() + Math.random() * 100000));
  renderMessages([]);
  nodes.traceText.textContent = "new session";
  nodes.input.focus();
});

setSession(state.sessionId);
checkHealth();
loadHistory();
