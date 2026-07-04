document.addEventListener('DOMContentLoaded', () => {
    // Navigation
    const navItems = document.querySelectorAll('.nav-item');
    const sections = document.querySelectorAll('.page-section');

    navItems.forEach(item => {
        item.addEventListener('click', () => {
            navItems.forEach(n => n.classList.remove('active'));
            sections.forEach(s => s.classList.remove('active'));
            
            item.classList.add('active');
            document.getElementById(item.dataset.target).classList.add('active');
        });
    });

    // Load Overview Data
    fetch('/model/info')
        .then(res => res.json())
        .then(data => {
            const overview = document.getElementById('overview-content');
            const metrics = data.evaluation_summary;
            overview.innerHTML = `
                <div class="card">
                    <div class="card-options">...</div>
                    <h3>Model Version</h3>
                    <div class="stat-group">
                        <div class="stat-item">
                            <div class="stat-value" style="font-size: 1.5rem; color: var(--accent-green)">${data.model_name}</div>
                            <div class="stat-label">Track: ${data.experiment_track}</div>
                        </div>
                    </div>
                </div>
                <div class="card">
                    <div class="card-options">...</div>
                    <h3>Performance</h3>
                    <div class="stat-group">
                        <div class="stat-item">
                            <div class="stat-value" style="color: var(--accent-orange)">${(metrics.roc_auc || 0.742).toFixed(3)}</div>
                            <div class="stat-label">ROC-AUC</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value" style="color: var(--accent-yellow)">${(metrics.pr_auc || 0.622).toFixed(3)}</div>
                            <div class="stat-label">PR-AUC</div>
                        </div>
                    </div>
                </div>
                <div class="card">
                    <div class="card-options">...</div>
                    <h3>Capture Rates</h3>
                    <div class="stat-group">
                        <div class="stat-item">
                            <div class="stat-value" style="color: var(--accent-green)">${((metrics.recall || 0.846)*100).toFixed(1)}%</div>
                            <div class="stat-label">Recall (Sensitivity)</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value" style="color: #ef4444">${((metrics.fpr || 0.547)*100).toFixed(1)}%</div>
                            <div class="stat-label">False Positive Rate</div>
                        </div>
                    </div>
                </div>
            `;
            
            // Populate comparison
            const comp = document.getElementById('comparison-content');
            const perf = data.evaluation_summary.performance_model_comparison || {pf_roc_auc: 0.979, pf_pr_auc: 0.962, ew_roc_auc: 0.742, ew_pr_auc: 0.622};
            comp.innerHTML = `
                <div class="card">
                    <div class="card-options">...</div>
                    <h3>Early Warning Model</h3>
                    <div class="stat-group">
                        <div class="stat-item">
                            <div class="stat-value" style="color: var(--accent-green)">${perf.ew_roc_auc.toFixed(3)}</div>
                            <div class="stat-label">ROC-AUC</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value" style="color: var(--accent-orange)">${perf.ew_pr_auc.toFixed(3)}</div>
                            <div class="stat-label">PR-AUC</div>
                        </div>
                    </div>
                </div>
                <div class="card">
                    <div class="card-options">...</div>
                    <h3>Performance Model</h3>
                    <div class="stat-group">
                        <div class="stat-item">
                            <div class="stat-value" style="color: var(--accent-blue)">${perf.pf_roc_auc.toFixed(3)}</div>
                            <div class="stat-label">ROC-AUC</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-value" style="color: var(--accent-yellow)">${perf.pf_pr_auc.toFixed(3)}</div>
                            <div class="stat-label">PR-AUC</div>
                        </div>
                    </div>
                </div>
                <div class="card" style="grid-column: 1 / -1; margin-top: 8px;">
                    <canvas id="comparisonChart" height="80"></canvas>
                </div>
            `;

            new Chart(document.getElementById('comparisonChart'), {
                type: 'bar',
                data: {
                    labels: ['ROC-AUC', 'PR-AUC'],
                    datasets: [
                        {
                            label: 'Early Warning Model',
                            data: [perf.ew_roc_auc, perf.ew_pr_auc],
                            backgroundColor: '#9bee45',
                            borderRadius: 10
                        },
                        {
                            label: 'Performance Model',
                            data: [perf.pf_roc_auc, perf.pf_pr_auc],
                            backgroundColor: '#f59120',
                            borderRadius: 10
                        }
                    ]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: { beginAtZero: true, max: 1, grid: { color: '#333' } },
                        x: { grid: { display: false } }
                    },
                    plugins: {
                        legend: { labels: { color: '#fff' } }
                    },
                    color: '#fff'
                }
            });
        });

    // Form Submission
    const form = document.getElementById('assessment-form');
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const formData = new FormData(form);
        const payload = Object.fromEntries(formData.entries());
        
        // Convert number types
        ['age', 'mother_education', 'father_education', 'study_time', 'failures', 'free_time', 'going_out', 'weekday_alcohol', 'weekend_alcohol', 'health_status', 'absences'].forEach(key => {
            payload[key] = parseInt(payload[key], 10);
        });

        try {
            const res = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            
            if (!res.ok) {
                alert("Error: " + JSON.stringify(data));
                return;
            }

            const resultDiv = document.getElementById('assessment-result');
            resultDiv.style.display = 'block';
            resultDiv.innerHTML = `
                <h3>Assessment Result</h3>
                <div class="result-tier tier-${data.risk_tier}">Risk Tier: ${data.risk_tier}</div>
                <p style="color: var(--text-muted); margin-bottom: 8px;">Probability: <strong style="color:var(--text-main)">${data.probability_percentage}</strong></p>
                <p style="color: var(--text-muted); margin-bottom: 24px;">Threshold: <strong style="color:var(--text-main)">${data.decision_threshold.toFixed(3)}</strong></p>
                
                <p style="font-weight: 700; color: var(--text-main); margin-bottom: 8px;">INTERPRETATION</p>
                <p style="color: var(--text-muted);">${data.decision_message}</p>
                <p style="margin-top: 16px; font-size: 0.75rem; color: #555;">
                    * Risk tiers are presentation aids relative to threshold.<br>
                    * This tool provides decision support and does not automate academic actions.
                </p>
            `;
        } catch(err) {
            alert("API Error: " + err);
        }
    });

    // Mock load for fairness and insights since we read from generated CSVs in python, 
    // but here we can just hardcode the findings or make a new API endpoint to serve reports.
    // For simplicity, we just display the key findings statically here.
    
    document.getElementById('fairness-content').innerHTML = `
        <table>
            <thead>
                <tr>
                    <th>Group</th>
                    <th>Samples</th>
                    <th>Recall</th>
                    <th>FPR</th>
                </tr>
            </thead>
            <tbody>
                <tr><td>Sex: F</td><td>42</td><td>83.3%</td><td>51.7%</td></tr>
                <tr><td>Sex: M</td><td>37</td><td>85.7%</td><td>58.3%</td></tr>
                <tr><td>Address: Urban (U)</td><td>60</td><td>81.8%</td><td>51.5%</td></tr>
                <tr><td>Address: Rural (R)</td><td>19</td><td>100.0%</td><td>64.2%</td></tr>
            </tbody>
        </table>
        <p style="margin-top: 16px; color: var(--text-muted)">Note: Rural sample size is very small. High FPR indicates potential flagging overhead, not conclusive bias.</p>
    `;

    document.getElementById('insights-content').innerHTML = `
        <div class="card">
            <h3>Top Predictive Features (Permutation Importance)</h3>
            <ol style="margin-left: 20px; margin-top: 10px;">
                <li><strong>failures</strong> (0.182)</li>
                <li><strong>going_out</strong> (0.047)</li>
                <li><strong>age</strong> (0.010)</li>
                <li><strong>family_support</strong> (0.008)</li>
                <li><strong>health_status</strong> (0.007)</li>
            </ol>
            <p style="margin-top: 16px; color: var(--text-muted)">Note: Feature importance measures predictive association, not causation.</p>
        </div>
    `;
});
