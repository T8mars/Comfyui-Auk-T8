import { app } from "../../../scripts/app.js";

let guides = {};
const guideUrl = new URL("../task_guides.json", import.meta.url);
fetch(guideUrl)
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
                minHeight: "112px",
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
            });
            guideWidget.serialize = false;
            guideWidget.options = { ...(guideWidget.options || {}), serialize: false };

            this.aukRefreshTaskGuide = () => renderGuide(container, taskWidget.value);
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
