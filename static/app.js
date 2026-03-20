// app.js -- upload handler + analytics charts

/* ============================================================
   UPLOAD / ANALYZE (original logic preserved)
   ============================================================ */
const analyzeBtn    = document.getElementById('analyzeBtn');
const videoFileInput = document.getElementById('fileInput');
const statusEl      = document.getElementById('status');
const spinner       = document.getElementById('spinner');
const predictionEl  = document.getElementById('prediction');
const explanationEl = document.getElementById('explanation');
const gradcamsEl    = document.getElementById('gradcams');
const audioMapEl    = document.getElementById('audioMap');
const videoPlayer   = document.getElementById('videoPlayer');

function setStatus(text, showSpinner = true) {
    if (!statusEl) return;
    statusEl.innerText = text;
    if (spinner) spinner.classList.toggle('hidden', !showSpinner);
}

function resetResults() {
    if (predictionEl)  predictionEl.innerText = '';
    if (explanationEl) explanationEl.value    = '';
    if (gradcamsEl)    gradcamsEl.innerHTML   = '';
    if (audioMapEl)    audioMapEl.innerHTML   = '';
    if (videoPlayer)   videoPlayer.src        = '';
}

if (analyzeBtn) {
    analyzeBtn.addEventListener('click', async () => {
        if (!videoFileInput || !videoFileInput.files || videoFileInput.files.length === 0) {
            alert('Please choose a video file first.');
            return;
        }
        resetResults();
        const file = videoFileInput.files[0];
        setStatus('🔍 Analyzing video... Please wait (may take 30–120s)...', true);

        const formData = new FormData();
        formData.append('video', file);

        try {
            const res = await fetch('/analyze', { method: 'POST', body: formData });

            if (!res.ok) {
                const err = await res.json().catch(() => ({ error: 'Unknown error' }));
                setStatus('⚠️ Error: ' + (err.error || res.statusText), false);
                return;
            }

            const data = await res.json();
            setStatus('✅ Analysis complete', false);

            if (predictionEl)  predictionEl.innerText = data.prediction  || 'N/A';
            if (explanationEl) explanationEl.value    = data.explanation || '';

            if (data.gradcams && data.gradcams.length) {
                data.gradcams.forEach(p => {
                    const img = document.createElement('img');
                    img.src = p;
                    gradcamsEl.appendChild(img);
                });
            }

            if (data.audio_heatmap) {
                const aimg = document.createElement('img');
                aimg.src = data.audio_heatmap;
                audioMapEl.appendChild(aimg);
            }

            if (data.video_preview) {
                videoPlayer.src = data.video_preview;
                videoPlayer.load();
            }

        } catch (e) {
            console.error(e);
            setStatus('⚠️ Unexpected error: ' + e.message, false);
        }
    });
}

/* ============================================================
   ANALYTICS -- only runs on result page (canvas elements exist)
   ============================================================ */

/* Guard: variables are injected by result.html; skip if absent */
if (typeof TEMPORAL_PROBS !== 'undefined') {

    /* ----------------------------------------------------------
       THEME PALETTE
    ---------------------------------------------------------- */
    const PAL = {
        crimson:   'rgba(192,81,58,1)',
        crimsonFg: 'rgba(192,81,58,0.85)',
        threshold: 'rgba(176,128,96,0.75)',
        gridLine:  'rgba(220,195,172,0.55)',
        tick:      '#9a7060',
        font:      'Georgia, serif',
        bars:      [
            'rgba(192,98,58,0.82)',
            'rgba(176,80,56,0.82)',
            'rgba(196,130,72,0.82)',
            'rgba(122,171,128,0.82)',
            'rgba(162,110,80,0.82)',
        ],
        barsHover: [
            'rgba(192,98,58,1)',
            'rgba(176,80,56,1)',
            'rgba(196,130,72,1)',
            'rgba(122,171,128,1)',
            'rgba(162,110,80,1)',
        ],
    };

    Chart.defaults.font.family = PAL.font;
    Chart.defaults.color       = PAL.tick;

    /* ----------------------------------------------------------
       1. TEMPORAL ANALYSIS -- animated line chart
    ---------------------------------------------------------- */
    (function buildTemporalChart() {
        const canvas = document.getElementById('temporalChart');
        if (!canvas) return;

        let probs  = Array.isArray(TEMPORAL_PROBS)  && TEMPORAL_PROBS.length  ? TEMPORAL_PROBS  : [];
        let ranges = Array.isArray(TEMPORAL_RANGES) && TEMPORAL_RANGES.length ? TEMPORAL_RANGES : [];

        /* Demo fallback */
        if (!probs.length) {
            probs  = Array.from({ length: 20 }, () => +(Math.random() * 0.7 + 0.1).toFixed(3));
            ranges = probs.map((_, i) => +(i * 1.5).toFixed(1));
        }

        const labels = ranges.length === probs.length
            ? ranges.map(t => t + 's')
            : probs.map((_, i) => (i * (probs.length > 1 ? 30 / (probs.length - 1) : 0)).toFixed(1) + 's');

        /* External tooltip */
        const tooltipEl = document.createElement('div');
        tooltipEl.className = 'analytics-tooltip';
        tooltipEl.style.opacity = '0';
        canvas.parentElement.style.position = 'relative';
        canvas.parentElement.appendChild(tooltipEl);

        function externalTooltip({ chart, tooltip }) {
            if (tooltip.opacity === 0) { tooltipEl.style.opacity = '0'; return; }
            const dp = tooltip.dataPoints[0];
            tooltipEl.innerHTML =
                '<div class="tt-time">⏱ ' + dp.label + '</div>' +
                '<div class="tt-prob">Forgery: <strong>' + (dp.raw * 100).toFixed(1) + '%</strong></div>';
            const x = chart.canvas.offsetLeft + tooltip.caretX;
            const y = chart.canvas.offsetTop  + tooltip.caretY;
            tooltipEl.style.opacity = '1';
            tooltipEl.style.left    = (x + 12) + 'px';
            tooltipEl.style.top     = (y - 20) + 'px';
        }

        const ctx  = canvas.getContext('2d');
        const grad = ctx.createLinearGradient(0, 0, 0, 260);
        grad.addColorStop(0, 'rgba(192,81,58,0.18)');
        grad.addColorStop(1, 'rgba(192,81,58,0.0)');

        new Chart(ctx, {
            type: 'line',
            data: {
                labels,
                datasets: [
                    {
                        label:               'Forgery Probability',
                        data:                probs,
                        borderColor:         PAL.crimsonFg,
                        backgroundColor:     grad,
                        borderWidth:         2.5,
                        pointRadius:         4,
                        pointHoverRadius:    6,
                        pointBackgroundColor: PAL.crimson,
                        pointBorderColor:    '#fff8f0',
                        pointBorderWidth:    2,
                        tension:             0.42,
                        fill:                true,
                    },
                    {
                        label:       'Threshold',
                        data:        new Array(probs.length).fill(0.5),
                        borderColor: PAL.threshold,
                        borderWidth: 1.5,
                        borderDash:  [6, 5],
                        pointRadius: 0,
                        tension:     0,
                        fill:        false,
                    },
                ],
            },
            options: {
                responsive:          true,
                maintainAspectRatio: false,
                animation: { duration: 1200, easing: 'easeInOutQuart' },
                interaction: { mode: 'index', intersect: false },
                plugins: {
                    legend:  { display: false },
                    tooltip: {
                        enabled:  false,
                        external: externalTooltip,
                        filter:   item => item.datasetIndex === 0,
                    },
                },
                scales: {
                    x: {
                        grid:  { color: PAL.gridLine, drawBorder: false },
                        ticks: { color: PAL.tick, font: { family: PAL.font, size: 12 }, maxTicksLimit: 10 },
                        title: { display: true, text: 'Time (seconds)', color: PAL.tick, font: { family: PAL.font, size: 12, style: 'italic' } },
                    },
                    y: {
                        min:   0,
                        max:   1,
                        grid:  { color: PAL.gridLine, drawBorder: false },
                        ticks: { color: PAL.tick, font: { family: PAL.font, size: 12 }, callback: v => (v * 100).toFixed(0) + '%', stepSize: 0.25 },
                        title: { display: true, text: 'Forgery Probability', color: PAL.tick, font: { family: PAL.font, size: 12, style: 'italic' } },
                    },
                },
            },
        });
    }());

    /* ----------------------------------------------------------
       2. AUDIO & VISUAL ANALYTICS -- animated bar chart
    ---------------------------------------------------------- */
    (function buildAVChart() {
        const canvas = document.getElementById('avChart');
        if (!canvas) return;

        const entries = Object.entries(AV_SCORES);
        const labels  = entries.map(([k]) => k);
        const values  = entries.map(([, v]) => +Math.min(1, Math.max(0, v)).toFixed(3));

        new Chart(canvas.getContext('2d'), {
            type: 'bar',
            data: {
                labels,
                datasets: [{
                    label:                'Anomaly Score',
                    data:                 values,
                    backgroundColor:      PAL.bars,
                    hoverBackgroundColor: PAL.barsHover,
                    borderRadius:         8,
                    borderSkipped:        false,
                    borderWidth:          0,
                    barPercentage:        0.6,
                    categoryPercentage:   0.7,
                }],
            },
            options: {
                responsive:          true,
                maintainAspectRatio: false,
                animation: {
                    duration: 1100,
                    easing:   'easeOutQuart',
                    delay: ctx => ctx.type === 'data' && ctx.mode === 'default' ? ctx.dataIndex * 80 : 0,
                },
                plugins: {
                    legend:  { display: false },
                    tooltip: {
                        backgroundColor: '#fff8f0',
                        borderColor:     '#e0c8b0',
                        borderWidth:     1.5,
                        titleColor:      '#a5531c',
                        bodyColor:       '#4a3b39',
                        titleFont:       { family: PAL.font, size: 13, weight: 'bold' },
                        bodyFont:        { family: PAL.font, size: 13 },
                        padding:         10,
                        callbacks:       { label: item => 'Score: ' + (item.raw * 100).toFixed(1) + '%' },
                    },
                },
                scales: {
                    x: {
                        grid:  { display: false },
                        ticks: { color: PAL.tick, font: { family: PAL.font, size: 12 } },
                    },
                    y: {
                        min:   0,
                        max:   1,
                        grid:  { color: PAL.gridLine, drawBorder: false },
                        ticks: { color: PAL.tick, font: { family: PAL.font, size: 12 }, callback: v => (v * 100).toFixed(0) + '%', stepSize: 0.25 },
                    },
                },
            },
        });
    }());

    /* ----------------------------------------------------------
       3. FRAME INSPECTION -- Circular SVG gauge dials
    ---------------------------------------------------------- */
    (function buildGauges() {
        const container = document.getElementById('gaugesRow');
        if (!container) return;

        if (!Array.isArray(FRAME_DATA) || !FRAME_DATA.length) {
            container.innerHTML = '<p class="muted-text">No frame-level data available.</p>';
            return;
        }

        var R    = 30;
        var CIRC = 2 * Math.PI * R;

        function gaugeColor(v) {
            if (v >= 0.65) return { stroke: '#7aab80', text: '#4a8a52' };
            if (v >= 0.35) return { stroke: '#c9923a', text: '#a06820' };
            return { stroke: '#d98a82', text: '#b04840' };
        }

        FRAME_DATA.forEach(function(frame, idx) {
            var val    = Math.min(1, Math.max(0, frame.value));
            var pct    = (val * 100).toFixed(1);
            var col    = gaugeColor(val);
            var fillId = 'gf' + idx;

            var wrap = document.createElement('div');
            wrap.className = 'gauge-wrap';
            wrap.style.animationDelay = (idx * 70) + 'ms';
            wrap.innerHTML =
                '<svg class="gauge-svg" viewBox="0 0 72 72">' +
                  '<circle class="gauge-track" cx="36" cy="36" r="' + R + '"/>' +
                  '<circle class="gauge-fill" id="' + fillId + '" cx="36" cy="36" r="' + R + '"' +
                    ' stroke="' + col.stroke + '"' +
                    ' stroke-dasharray="' + CIRC + '"' +
                    ' stroke-dashoffset="' + CIRC + '"/>' +
                  '<text class="gauge-pct" x="36" y="33" fill="' + col.text + '">' + pct + '%</text>' +
                  '<text class="gauge-sublabel" x="36" y="44">' + frame.label + '</text>' +
                '</svg>';

            container.appendChild(wrap);
        });

        /* Animate stroke after paint */
        requestAnimationFrame(function() {
            requestAnimationFrame(function() {
                FRAME_DATA.forEach(function(frame, idx) {
                    var val    = Math.min(1, Math.max(0, frame.value));
                    var offset = CIRC * (1 - val);
                    var el     = document.getElementById('gf' + idx);
                    if (el) el.style.strokeDashoffset = offset;
                });
            });
        });
    }());

} // end analytics guard