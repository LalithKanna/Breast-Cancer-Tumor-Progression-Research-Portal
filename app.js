document.getElementById("uploadForm").addEventListener("submit", async (e) => {
    e.preventDefault();

    const formData = new FormData();
    formData.append("sequence", document.getElementById("seqFile").files[0]);
    formData.append("patient_data", document.getElementById("staticFile").files[0]);

    const res = await fetch("/api/analyze", {
        method: "POST",
        body: formData
    });

    const data = await res.json();

    if (!res.ok) {
        alert(data.error || "Analysis failed");
        return;
    }

    // Show results container
    document.getElementById("initialMessage").classList.add("hidden");
    document.getElementById("resultsContainer").classList.remove("hidden");

    // Metrics
    document.getElementById("deathProb").textContent =
        (data.death_prob * 100).toFixed(1) + "%";
    document.getElementById("survivalProb").textContent =
        (data.survival_prob * 100).toFixed(1) + "%";
    document.getElementById("recurrenceProb").textContent =
        (data.recurrence_prob * 100).toFixed(1) + "%";

    // Survival curves
    Plotly.newPlot("osChart", [{
        x: data.times,
        y: data.os_surv,
        mode: "lines",
        name: "Overall Survival"
    }]);

    Plotly.newPlot("rfsChart", [{
        x: data.times,
        y: data.rfs_surv,
        mode: "lines",
        name: "Relapse-Free Survival"
    }]);

    // Risk table
    const riskTable = document.getElementById("riskTable");
    riskTable.innerHTML = "";
    data.risk_estimates.forEach(r => {
        riskTable.innerHTML += `
            <tr>
                <td>${r["Time Horizon"]}</td>
                <td>${(r["Survival Probability"] * 100).toFixed(1)}%</td>
                <td>${(r["Death Probability"] * 100).toFixed(1)}%</td>
                <td>${(r["Recurrence Probability"] * 100).toFixed(1)}%</td>
            </tr>
        `;
    });

    // Explainability table
    const explainTable = document.getElementById("explainabilityTable");
    explainTable.innerHTML = "";
    data.explainability.features.forEach(f => {
        explainTable.innerHTML += `
            <tr>
                <td>${f.name}</td>
                <td>${f.value}</td>
                <td>${f.shap_value.toFixed(3)}</td>
            </tr>
        `;
    });

    // SHAP bar chart
    Plotly.newPlot("shpChart", [{
        x: data.explainability.features.map(f => f.shap_value),
        y: data.explainability.features.map(f => f.name),
        type: "bar",
        orientation: "h"
    }]);

    // Raw JSON
    document.getElementById("jsonOutput").textContent =
        JSON.stringify(data.explainability, null, 2);

    // Improvements
    const impTable = document.getElementById("improvementTable");
    impTable.innerHTML = "";
    data.improvements.forEach(i => {
        impTable.innerHTML += `
            <tr>
                <td>${i.Feature}</td>
                <td>${i["Hypothetical Adjustment"]}</td>
                <td>${(i["5-Year Survival Gain"] * 100).toFixed(1)}%</td>
            </tr>
        `;
    });
});
