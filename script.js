
const API = "http://127.0.0.1:8000";

let datasetId = null;
let selectedFeatures = [];
let currentWeights = {};
let proposedWeights = null;

async function apiRequest(url, options = {}) {
    const response = await fetch(`${API}${url}`, options);
    const data = await response.json();

    if (!response.ok) {
        throw new Error(data.detail || "Something went wrong.");
    }

    return data;
}

async function uploadDataset() {
    const file = document.getElementById("fileInput").files[0];

    if (!file) {
        alert("Please select a CSV file.");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    document.getElementById("uploadStatus").innerText =
        "Uploading dataset...";

    try {
        const data = await apiRequest("/upload", {
            method: "POST",
            body: formData
        });

        datasetId = data.dataset_id;
        selectedFeatures = [];
        currentWeights = {};
        proposedWeights = null;

        showProfile(data.profile);

        document.getElementById("uploadStatus").innerText =
            "Dataset uploaded successfully!";

        document.getElementById("resultsSection").hidden = true;

        const aiSection = document.getElementById("aiSection");
        if (aiSection) aiSection.hidden = true;

    } catch (error) {
        alert(error.message);
        document.getElementById("uploadStatus").innerText =
            "Upload failed.";
    }
}

function showProfile(profile) {
    document.getElementById("profileSection").hidden = false;

    document.getElementById("stats").innerHTML = `
        <div class="stat"><b>${profile.rows}</b><br>Rows</div>
        <div class="stat"><b>${profile.total_columns}</b><br>Columns</div>
        <div class="stat"><b>${profile.missing_cells}</b><br>Missing Cells</div>
        <div class="stat"><b>${profile.duplicate_rows}</b><br>Duplicate Rows</div>
    `;

    document.getElementById("schemaBody").innerHTML =
        profile.columns.map(col => `
            <tr>
                <td>${col.column}</td>
                <td>${col.type}</td>
                <td>${col.missing}</td>
                <td>${col.unique}</td>
                <td>${col.role}</td>
            </tr>
        `).join("");

    document.getElementById("featureList").innerHTML =
        profile.features.map(feature => `
            <label>
                <input type="checkbox"
                    value="${feature.column}"
                    ${feature.type === "Numeric" ? "checked" : ""}>
                ${feature.column}
            </label><br>
        `).join("");
}

async function runSegmentation() {
    if (!datasetId) {
        alert("Upload a dataset first.");
        return;
    }

    selectedFeatures = [
        ...document.querySelectorAll("#featureList input:checked")
    ].map(input => input.value);

    if (!selectedFeatures.length) {
        alert("Select at least one feature.");
        return;
    }

    selectedFeatures.forEach(feature => {
        const slider = document.getElementById("weight_" + feature);

        if (slider) {
            currentWeights[feature] = Number(slider.value);
        } else if (currentWeights[feature] === undefined) {
            currentWeights[feature] = 1;
        }
    });

    const weights = {};
    selectedFeatures.forEach(feature => {
        weights[feature] = currentWeights[feature] ?? 1;
    });

    try {
        const data = await apiRequest("/segment", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                dataset_id: datasetId,
                selected_features: selectedFeatures,
                weights: weights,
                n_clusters: Number(
                    document.getElementById("clusterCount").value
                )
            })
        });

        showResults(data);

        const aiSection = document.getElementById("aiSection");
        if (aiSection) aiSection.hidden = false;

        const analysisOutput = document.getElementById("aiAnalysis");
        if (analysisOutput) {
            analysisOutput.innerText =
                "Segmentation updated. Run AI Analysis again to refresh the explanation.";
        }

        proposedWeights = null;
        const proposal = document.getElementById("proposal");
        if (proposal) proposal.hidden = true;

    } catch (error) {
        alert(error.message);
    }
}

function showResults(data) {
    document.getElementById("resultsSection").hidden = false;

    document.getElementById("results").innerHTML =
        data.methods.map(method => `
            <div class="result-card">
                <h3>${method.method}</h3>
                <h2>Silhouette Score: ${method.silhouette ?? "N/A"}</h2>
                <p>Clusters: ${method.clusters}</p>
                <p>Cluster sizes: ${JSON.stringify(method.cluster_sizes)}</p>
            </div>
        `).join("");

    document.getElementById("results").innerHTML += `
        <h3>Best method by Silhouette Score:
        ${data.best_method || "Not available"}</h3>
    `;

    document.getElementById("weights").innerHTML =
        selectedFeatures.map(feature => `
            <div class="weight-row">
                <label>${feature}</label>
                <input
                    type="range"
                    id="weight_${feature}"
                    min="0"
                    max="2"
                    step="0.1"
                    value="${currentWeights[feature] ?? 1}"
                    oninput="updateWeight('${feature}', this.value)">
                <span id="value_${feature}">
                    ${currentWeights[feature] ?? 1}
                </span>
            </div>
        `).join("");
}

function updateWeight(feature, value) {
    currentWeights[feature] = Number(value);

    const display = document.getElementById("value_" + feature);
    if (display) display.innerText = value;
}

// AI analysis of actual clustering results
async function runAIAnalysis() {
    if (!datasetId) return;

    const output = document.getElementById("aiAnalysis");
    output.innerText = "AI is analysing the clustering results...";

    try {
        const data = await apiRequest(`/analyze/${datasetId}`);

        output.innerText = data.analysis || "No analysis was returned.";

    } catch (error) {
        output.innerText = "AI analysis failed: " + error.message;
    }
}

// Convert a business instruction into proposed feature weights
async function suggestChanges() {
    const instruction = document.getElementById("aiInstruction").value.trim();

    if (!instruction) {
        alert("Please enter an instruction.");
        return;
    }

    if (!datasetId) {
        alert("Upload a dataset and run segmentation first.");
        return;
    }

    const output = document.getElementById("proposalOutput");
    output.innerText = "AI is preparing your proposed changes...";

    document.getElementById("proposal").hidden = false;
    document.getElementById("approveChanges").disabled = true;

    try {
        const weights = {};
        selectedFeatures.forEach(feature => {
            weights[feature] = currentWeights[feature] ?? 1;
        });

        const data = await apiRequest("/suggest-adjustments", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                dataset_id: datasetId,
                instruction: instruction,
                selected_features: selectedFeatures,
                current_weights: weights
            })
        });

        proposedWeights = data.proposed_weights;

        output.innerText =
            (data.explanation || "Proposed changes:") +
            "\n\n" +
            Object.entries(proposedWeights)
                .map(([feature, weight]) => `${feature}: ${weight}`)
                .join("\n");

        document.getElementById("approveChanges").disabled = false;

    } catch (error) {
        proposedWeights = null;
        output.innerText = "Could not prepare changes: " + error.message;
    }
}

// Apply only after the user explicitly approves
async function approveChanges() {
    if (!proposedWeights) return;

    Object.entries(proposedWeights).forEach(([feature, weight]) => {
        currentWeights[feature] = Number(weight);
    });

    Object.entries(proposedWeights).forEach(([feature, weight]) => {
        const slider = document.getElementById("weight_" + feature);
        const display = document.getElementById("value_" + feature);

        if (slider) slider.value = weight;
        if (display) display.innerText = weight;
    });

    proposedWeights = null;

    document.getElementById("proposal").hidden = true;

    await runSegmentation();
}

async function downloadReport() {
    if (!datasetId) return;

    try {
        const response = await fetch(`${API}/report/${datasetId}`);

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || "Report download failed.");
        }

        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");

        link.href = url;
        link.download = "hcp_segmentation_report.xlsx";
        link.click();

        URL.revokeObjectURL(url);

    } catch (error) {
        alert(error.message);
    }
}

// Create the AI interface without requiring changes to index.html
function addAISection() {
    if (document.getElementById("aiSection")) return;

    const section = document.createElement("section");
    section.id = "aiSection";
    section.hidden = true;

    section.innerHTML = `
        <hr>
        <h2>4. AI-Powered Business Analysis</h2>

        <p>
            Understand the segmentation results, compare the
            three techniques and get business recommendations.
        </p>

        <button onclick="runAIAnalysis()">
            Generate AI Analysis
        </button>

        <div id="aiAnalysis"
             style="white-space:pre-wrap; margin-top:15px;">
        </div>

        <hr>

        <h3>Refine Segmentation Using Natural Language</h3>

        <p>
            Describe how you want the segmentation to change.
        </p>

        <textarea
            id="aiInstruction"
            rows="3"
            style="width:100%;"
            placeholder="Example: Give more importance to prescription volume than engagement.">
        </textarea>

        <br><br>

        <button onclick="suggestChanges()">
            Ask AI to Suggest Changes
        </button>

        <div id="proposal" hidden style="margin-top:15px;">
            <h3>Review Proposed Changes</h3>

            <div id="proposalOutput"
                 style="white-space:pre-wrap;"></div>

            <br>

            <button id="approveChanges"
                    onclick="approveChanges()"
                    disabled>
                Approve Changes & Rerun Segmentation
            </button>
        </div>
    `;

    document.getElementById("resultsSection").appendChild(section);
}

document.addEventListener("DOMContentLoaded", addAISection);
