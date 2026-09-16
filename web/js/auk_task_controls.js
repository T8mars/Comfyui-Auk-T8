import { app } from "../../../scripts/app.js";

let guides = {};
const guideUrl = new URL("../task_guides.json", import.meta.url);
guideUrl.searchParams.set("v", "2.0.7");
fetch(guideUrl, { cache: "no-store" })
    .then((response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json();
    })
    .then((value) => {
        guides = value;
        for (const node of app.graph?._nodes || []) node.aukRefreshTaskGuide?.();
    })
    .catch((error) => console.error("AuK task guide failed to load", error));

function renderGuide(container, label) {
    const guide = guides[label];
    if (!guide) {
        container.textContent = "正在载入 AuK 官方任务要求…";
        return;
    }
    container.replaceChildren();
    const title = document.createElement("strong");
    title.textContent = `官方用法 · ${label}`;
    const requirement = document.createElement("div");
    requirement.textContent = guide.requirement;
    const fields = document.createElement("div");
    fields.textContent = `填写：${guide.primary_label}${guide.secondary_label ? `｜${guide.secondary_label}` : ""}`;
    const example = document.createElement("div");
    example.textContent = `示例：${guide.example}`;
    const note = document.createElement("div");
    note.textContent = guide.note;
    note.style.opacity = "0.78";
    container.append(title, requirement, fields, example, note);
}

app.registerExtension({
    name: "T8star.AuKTaskGuide",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "AuKGenerateEdit") return;
        const originalCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalCreated?.apply(this, arguments);
            const taskWidget = this.widgets?.find((widget) => widget.name === "task");
            if (!taskWidget || typeof this.addDOMWidget !== "function") return result;

            const container = document.createElement("div");
            Object.assign(container.style, {
                boxSizing: "border-box",
                width: "100%",
                height: "100%",
                minHeight: "0",
                overflow: "auto",
                padding: "10px 12px",
                color: "#1f2937",
                background: "#fff3f8",
                border: "1px solid #ff9dc8",
                borderRadius: "8px",
                fontSize: "12px",
                lineHeight: "1.55",
                whiteSpace: "normal",
            });
            const guideWidget = this.addDOMWidget("auk_task_guide", "div", container, {
                serialize: false,
                hideOnZoom: false,
                getMinHeight: () => 160,
                getHeight: () => 160,
            });
            guideWidget.serialize = false;
            guideWidget.options = { ...(guideWidget.options || {}), serialize: false };

            this.aukRefreshTaskGuide = () => {
                renderGuide(container, taskWidget.value);
                const durationWidget = this.widgets?.find((widget) => widget.name === "duration_mode");
                if (!durationWidget) return;
                const button = document.createElement("button");
                button.type = "button";
                button.textContent = "↻ 自动适配时长";
                button.title = "TTS 按目标文本估时；编辑按实际输入和任务规则计算，忽略连接的 Float。";
                Object.assign(button.style, {
                    marginTop: "6px", padding: "4px 10px", border: "1px solid #ff9dc8",
                    borderRadius: "6px", background: "#ffffff", color: "#b31765", cursor: "pointer",
                });
                button.onclick = (event) => {
                    event.stopPropagation();
                    const secondsWidget = this.widgets?.find((widget) => widget.name === "generation_seconds");
                    if (secondsWidget && (!Number.isFinite(Number(secondsWidget.value)) ||
                        Number(secondsWidget.value) < 0.2 || Number(secondsWidget.value) > 30)) {
                        secondsWidget.value = 3;
                    }
                    durationWidget.value = "自动适配（按任务规则）";
                    durationWidget.callback?.(durationWidget.value);
                    app.graph?.setDirtyCanvas(true, true);
                };
                container.prepend(button);
            };
            const originalCallback = taskWidget.callback;
            taskWidget.callback = (value, ...args) => {
                const callbackResult = originalCallback?.call(taskWidget, value, ...args);
                this.aukRefreshTaskGuide();
                return callbackResult;
            };
            this.aukRefreshTaskGuide();
            return result;
        };
    },
    loadedGraphNode(node) {
        node.aukRefreshTaskGuide?.();
    },
});
