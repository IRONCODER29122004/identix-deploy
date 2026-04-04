// Add this JavaScript code to landmark_index.html

// Deepfake drag and drop handlers (add after video drag-drop)
deepfakeUploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    deepfakeUploadArea.classList.add('dragover');
});

deepfakeUploadArea.addEventListener('dragleave', () => {
    deepfakeUploadArea.classList.remove('dragover');
});

deepfakeUploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    deepfakeUploadArea.classList.remove('dragover');
    selectedDeepfakeVideo = e.dataTransfer.files[0];
    if (selectedDeepfakeVideo) {
        deepfakeUploadArea.querySelector('.upload-text').textContent = selectedDeepfakeVideo.name;
        deepfakeAnalyzeBtn.style.display = 'block';
    }
});

// Deepfake Analysis Button
deepfakeAnalyzeBtn.addEventListener('click', async () => {
    if (!selectedDeepfakeVideo) return;

    const loadingText = document.getElementById('loadingText');
    const processingStatus = document.getElementById('processingStatus');
    loadingText.textContent = 'Analyzing video for deepfakes... This may take several minutes.';
    processingStatus.textContent = 'Running advanced detection algorithms...';
    loading.style.display = 'block';
    deepfakeResultsSection.style.display = 'none';
    errorMessage.style.display = 'none';
    deepfakeAnalyzeBtn.disabled = true;

    const formData = new FormData();
    formData.append('video', selectedDeepfakeVideo);
    formData.append('max_frames', document.getElementById('deepfakeMaxFrames').value);

    try {
        const response = await fetch('/detect_deepfake', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            const report = data.report;
            const deepfakeReport = document.getElementById('deepfakeReport');
            deepfakeReport.innerHTML = '';

            // Verdict Card
            const verdictColor = report.is_authentic === false ? '#ff4444' : 
                                 report.is_authentic === null ? '#ff9800' : '#4CAF50';
            
            const verdictIcon = report.is_authentic === false ? '⚠️' :
                               report.is_authentic === null ? '❓' : '✅';

            const verdictCard = document.createElement('div');
            verdictCard.style.cssText = `background: ${verdictColor}; color: white; padding: 30px; border-radius: 15px; margin-bottom: 30px; text-align: center;`;
            verdictCard.innerHTML = `
                <h2 style="margin: 0 0 15px 0; font-size: 2.5em;">${verdictIcon} ${report.verdict}</h2>
                <p style="margin: 0; font-size: 1.8em; font-weight: bold;">Authenticity Score: ${report.confidence}%</p>
                <p style="margin: 15px 0 0 0; font-size: 1.1em; opacity: 0.9;">
                    Analyzed ${report.video_info.analyzed_frames} frames from ${report.video_info.total_frames} total frames
                </p>
            `;
            deepfakeReport.appendChild(verdictCard);

            // Warnings Section
            if (report.warnings && report.warnings.length > 0) {
                const warningsDiv = document.createElement('div');
                warningsDiv.style.cssText = 'background: #fff3cd; border-left: 5px solid #ff9800; padding: 20px; border-radius: 10px; margin-bottom: 30px;';
                warningsDiv.innerHTML = `
                    <h3 style="color: #856404; margin: 0 0 15px 0;">⚠️ Detected Issues</h3>
                    <ul style="margin: 0; padding-left: 25px; color: #856404;">
                        ${report.warnings.map(w => `<li style="margin-bottom: 8px;">${w}</li>`).join('')}
                    </ul>
                `;
                deepfakeReport.appendChild(warningsDiv);
            }

            // Detailed Analysis Cards
            const analysisGrid = document.createElement('div');
            analysisGrid.style.cssText = 'display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 20px;';

            const analyses = [
                {
                    title: 'Temporal Consistency',
                    icon: '🎬',
                    score: report.details.temporal_consistency.score,
                    info: `${report.details.temporal_consistency.inconsistencies} irregular movements detected`
                },
                {
                    title: 'Boundary Artifacts',
                    icon: '🔍',
                    score: report.details.boundary_artifacts.score,
                    info: `${report.details.boundary_artifacts.avg_percentage.toFixed(1)}% suspicious pixels`
                },
                {
                    title: 'Blink Pattern',
                    icon: '👁️',
                    score: report.details.blink_analysis.score,
                    info: `${report.details.blink_analysis.blink_count} blinks detected`
                },
                {
                    title: 'Landmark Stability',
                    icon: '📍',
                    score: report.details.landmark_stability.score,
                    info: `Variance: ${report.details.landmark_stability.avg_variance.toFixed(2)}`
                }
            ];

            analyses.forEach(analysis => {
                const scoreColor = analysis.score >= 70 ? '#4CAF50' : 
                                  analysis.score >= 50 ? '#ff9800' : '#ff4444';
                
                const card = document.createElement('div');
                card.style.cssText = 'background: #f8f9ff; border-radius: 12px; padding: 20px; text-align: center; box-shadow: 0 4px 10px rgba(0,0,0,0.1);';
                card.innerHTML = `
                    <div style="font-size: 2.5em; margin-bottom: 10px;">${analysis.icon}</div>
                    <h4 style="margin: 0 0 10px 0; color: #667eea;">${analysis.title}</h4>
                    <div style="font-size: 2em; font-weight: bold; color: ${scoreColor}; margin-bottom: 8px;">${analysis.score.toFixed(1)}%</div>
                    <p style="margin: 0; color: #666; font-size: 0.9em;">${analysis.info}</p>
                `;
                analysisGrid.appendChild(card);
            });

            deepfakeReport.appendChild(analysisGrid);

            // Explanation Section
            const explanationDiv = document.createElement('div');
            explanationDiv.style.cssText = 'background: #e8f4f8; padding: 20px; border-radius: 10px; margin-top: 20px;';
            explanationDiv.innerHTML = `
                <h3 style="color: #0277bd; margin: 0 0 15px 0;">📊 How Detection Works</h3>
                <ul style="margin: 0; padding-left: 25px; color: #01579b;">
                    <li style="margin-bottom: 10px;"><strong>Temporal Consistency:</strong> Checks if facial landmarks move smoothly between frames</li>
                    <li style="margin-bottom: 10px;"><strong>Boundary Artifacts:</strong> Detects unnatural edges around the face boundary</li>
                    <li style="margin-bottom: 10px;"><strong>Blink Pattern:</strong> Analyzes if blinking appears natural and regular</li>
                    <li style="margin-bottom: 10px;"><strong>Landmark Stability:</strong> Measures jitter and instability in facial features</li>
                </ul>
            `;
            deepfakeReport.appendChild(explanationDiv);

            loading.style.display = 'none';
            deepfakeResultsSection.style.display = 'block';
        } else {
            throw new Error(data.error || 'Analysis failed');
        }
    } catch (error) {
        loading.style.display = 'none';
        errorMessage.textContent = `Error: ${error.message}`;
        errorMessage.style.display = 'block';
    } finally {
        deepfakeAnalyzeBtn.disabled = false;
        loadingText.textContent = 'Processing... This may take a few moments.';
        processingStatus.textContent = '';
    }
});

// Reset Deepfake
resetDeepfakeBtn.addEventListener('click', () => {
    selectedDeepfakeVideo = null;
    deepfakeInput.value = '';
    deepfakeUploadArea.querySelector('.upload-text').textContent = 'Drop video here to analyze for deepfakes';
    deepfakeAnalyzeBtn.style.display = 'none';
    deepfakeResultsSection.style.display = 'none';
    errorMessage.style.display = 'none';
});
