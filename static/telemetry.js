// Claude Code Proxy - Telemetry Viewer JavaScript

let currentTrace = null;
let currentView = 'timeline';
let autoRefreshInterval = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    refreshTraces();
    setupAutoRefresh();
});

function setupAutoRefresh() {
    const checkbox = document.getElementById('auto-refresh');
    if (checkbox.checked) {
        autoRefreshInterval = setInterval(refreshTraces, 5000);
    } else {
        clearInterval(autoRefreshInterval);
    }
    
    checkbox.addEventListener('change', () => {
        if (checkbox.checked) {
            autoRefreshInterval = setInterval(refreshTraces, 5000);
        } else {
            clearInterval(autoRefreshInterval);
        }
    });
}

async function refreshTraces() {
    try {
        const response = await fetch('/telemetry/traces');
        const data = await response.json();
        
        updateStats(data.stats);
        updateTraceSelector(data.traces);
        
        // If a trace is selected, refresh it
        const selector = document.getElementById('trace-selector');
        if (selector.value) {
            loadTrace(selector.value);
        }
    } catch (error) {
        console.error('Failed to refresh traces:', error);
    }
}

function updateStats(stats) {
    document.getElementById('total-traces').textContent = stats.total_traces;
    document.getElementById('total-spans').textContent = stats.total_spans;
    document.getElementById('avg-duration').textContent = stats.avg_duration.toFixed(2) + 'ms';
    document.getElementById('error-rate').textContent = stats.error_rate.toFixed(1) + '%';
}

function updateTraceSelector(traces) {
    const selector = document.getElementById('trace-selector');
    const currentValue = selector.value;
    
    selector.innerHTML = '<option value="">Select a trace...</option>';
    
    Object.entries(traces).forEach(([traceId, spans]) => {
        const option = document.createElement('option');
        option.value = traceId;
        
        const rootSpan = spans.find(s => !s.parent_span_id) || spans[0];
        const timestamp = new Date(rootSpan.start_time * 1000).toLocaleString();
        const duration = rootSpan.duration_ms ? `${rootSpan.duration_ms.toFixed(2)}ms` : 'ongoing';
        
        option.textContent = `${rootSpan.name} - ${timestamp} (${duration})`;
        
        if (traceId === currentValue) {
            option.selected = true;
        }
        
        selector.appendChild(option);
    });
}

function loadTrace(traceId) {
    if (!traceId) {
        currentTrace = null;
        clearVisualization();
        return;
    }
    
    fetch(`/telemetry/trace/${traceId}`)
        .then(response => response.json())
        .then(trace => {
            currentTrace = trace;
            renderCurrentView();
        })
        .catch(error => console.error('Failed to load trace:', error));
}

function switchView(view) {
    currentView = view;
    
    // Update tabs
    document.querySelectorAll('.tab').forEach(tab => {
        tab.classList.remove('active');
    });
    event.target.classList.add('active');
    
    // Hide all views
    document.querySelectorAll('.view-content').forEach(content => {
        content.style.display = 'none';
    });
    
    // Show selected view
    document.getElementById(`${view}-view`).style.display = 'block';
    
    renderCurrentView();
}

function renderCurrentView() {
    if (!currentTrace) return;
    
    switch (currentView) {
        case 'timeline':
            renderTimeline();
            break;
        case 'events':
            renderEvents();
            break;
        case 'json':
            renderJSON();
            break;
    }
}

function renderTimeline() {
    const container = document.getElementById('timeline');
    container.innerHTML = '';
    
    if (!currentTrace || currentTrace.length === 0) return;
    
    const margin = {top: 20, right: 20, bottom: 30, left: 200};
    const width = container.clientWidth - margin.left - margin.right;
    const height = Math.max(400, currentTrace.length * 40);
    
    const svg = d3.select(container)
        .append('svg')
        .attr('width', width + margin.left + margin.right)
        .attr('height', height + margin.top + margin.bottom);
    
    const g = svg.append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);
    
    // Calculate time bounds
    const minTime = d3.min(currentTrace, d => d.start_time);
    const maxTime = d3.max(currentTrace, d => d.end_time || d.start_time);
    
    const xScale = d3.scaleLinear()
        .domain([minTime, maxTime])
        .range([0, width]);
    
    const yScale = d3.scaleBand()
        .domain(currentTrace.map(d => d.span_id))
        .range([0, height])
        .padding(0.1);
    
    // Add x axis
    g.append('g')
        .attr('transform', `translate(0,${height})`)
        .call(d3.axisBottom(xScale)
            .tickFormat(d => `${((d - minTime) * 1000).toFixed(1)}ms`))
        .style('color', '#8892b0');
    
    // Create tooltip
    const tooltip = d3.select('body').append('div')
        .attr('class', 'timeline-tooltip')
        .style('opacity', 0);
    
    // Draw spans
    const spans = g.selectAll('.span')
        .data(currentTrace)
        .enter().append('g')
        .attr('class', 'span');
    
    spans.append('rect')
        .attr('x', d => xScale(d.start_time))
        .attr('y', d => yScale(d.span_id))
        .attr('width', d => {
            const endTime = d.end_time || maxTime;
            return Math.max(1, xScale(endTime) - xScale(d.start_time));
        })
        .attr('height', yScale.bandwidth())
        .attr('fill', d => d.status_code === 'ERROR' ? '#ff5555' : '#5a7fdb')
        .attr('opacity', 0.8)
        .on('mouseover', function(event, d) {
            tooltip.transition().duration(200).style('opacity', .9);
            tooltip.html(formatSpanTooltip(d))
                .style('left', (event.pageX + 10) + 'px')
                .style('top', (event.pageY - 28) + 'px');
        })
        .on('mouseout', function() {
            tooltip.transition().duration(500).style('opacity', 0);
        })
        .on('click', function(event, d) {
            showSpanDetails(d);
        });
    
    // Add span names
    spans.append('text')
        .attr('x', -5)
        .attr('y', d => yScale(d.span_id) + yScale.bandwidth() / 2)
        .attr('dy', '.35em')
        .attr('text-anchor', 'end')
        .text(d => d.name)
        .style('fill', '#e0e6ed')
        .style('font-size', '12px');
    
    // Add events as vertical lines
    currentTrace.forEach(span => {
        if (span.events && span.events.length > 0) {
            const eventLines = g.selectAll(`.event-${span.span_id}`)
                .data(span.events)
                .enter().append('line')
                .attr('x1', d => xScale(d.timestamp))
                .attr('y1', yScale(span.span_id))
                .attr('x2', d => xScale(d.timestamp))
                .attr('y2', yScale(span.span_id) + yScale.bandwidth())
                .attr('stroke', '#ff79c6')
                .attr('stroke-width', 2)
                .attr('stroke-dasharray', '2,2');
        }
    });
}


function renderEvents() {
    const container = document.getElementById('events-list');
    container.innerHTML = '';
    
    if (!currentTrace) return;
    
    // Collect all events from all spans
    const allEvents = [];
    currentTrace.forEach(span => {
        if (span.events) {
            span.events.forEach(event => {
                allEvents.push({
                    ...event,
                    span_id: span.span_id,
                    span_name: span.name
                });
            });
        }
    });
    
    // Sort by timestamp
    allEvents.sort((a, b) => a.timestamp - b.timestamp);
    
    allEvents.forEach(event => {
        const eventDiv = document.createElement('div');
        eventDiv.className = 'event-item';
        
        const headerDiv = document.createElement('div');
        headerDiv.className = 'event-header';
        
        const nameSpan = document.createElement('span');
        nameSpan.className = 'event-name';
        nameSpan.textContent = event.name;
        
        const timeSpan = document.createElement('span');
        timeSpan.className = 'event-time';
        timeSpan.textContent = new Date(event.timestamp * 1000).toLocaleTimeString();
        
        headerDiv.appendChild(nameSpan);
        headerDiv.appendChild(timeSpan);
        eventDiv.appendChild(headerDiv);
        
        // Add span info
        const spanInfo = document.createElement('div');
        spanInfo.style.color = '#8892b0';
        spanInfo.style.fontSize = '0.85em';
        spanInfo.textContent = `From span: ${event.span_name}`;
        eventDiv.appendChild(spanInfo);
        
        // Add attributes
        if (event.attributes && Object.keys(event.attributes).length > 0) {
            const attrDiv = document.createElement('div');
            attrDiv.style.marginTop = '10px';
            
            // Special handling for streaming chunks
            if (event.name === 'streaming_chunk_received') {
                const chunkIndex = event.attributes['proxy.streaming.chunk_index'];
                const chunkData = event.attributes['proxy.streaming.chunk_data'];
                
                // Show chunk index
                const indexTag = document.createElement('span');
                indexTag.className = 'attribute-tag';
                indexTag.textContent = `Chunk #${chunkIndex}`;
                attrDiv.appendChild(indexTag);
                
                // Parse and show chunk data
                if (chunkData) {
                    const dataDiv = document.createElement('div');
                    dataDiv.style.marginTop = '5px';
                    dataDiv.style.padding = '8px';
                    dataDiv.style.backgroundColor = '#0d1117';
                    dataDiv.style.borderRadius = '4px';
                    dataDiv.style.fontFamily = 'monospace';
                    dataDiv.style.fontSize = '11px';
                    dataDiv.style.maxHeight = '100px';
                    dataDiv.style.overflowY = 'auto';
                    
                    try {
                        if (chunkData.startsWith('data: ')) {
                            const jsonStr = chunkData.substring(6);
                            const parsed = JSON.parse(jsonStr);
                            dataDiv.innerHTML = syntaxHighlightJSON(JSON.stringify(parsed, null, 2));
                        } else {
                            dataDiv.textContent = chunkData;
                        }
                    } catch (e) {
                        dataDiv.textContent = chunkData;
                    }
                    
                    attrDiv.appendChild(dataDiv);
                }
            } else {
                // Regular attributes - use better formatting for long values
                attrDiv.className = 'attribute-section';
                
                Object.entries(event.attributes).forEach(([key, value]) => {
                    const itemDiv = document.createElement('div');
                    itemDiv.className = 'attribute-item';
                    
                    const keyDiv = document.createElement('div');
                    keyDiv.className = 'attribute-key';
                    keyDiv.textContent = key;
                    itemDiv.appendChild(keyDiv);
                    
                    const valueDiv = document.createElement('div');
                    valueDiv.className = 'attribute-value';
                    
                    // Format value based on type
                    if (typeof value === 'object' && value !== null) {
                        valueDiv.innerHTML = syntaxHighlightJSON(JSON.stringify(value, null, 2));
                    } else if (typeof value === 'string' && value.length > 100) {
                        // For long strings, make them more readable
                        valueDiv.textContent = value;
                    } else {
                        valueDiv.textContent = JSON.stringify(value);
                    }
                    
                    itemDiv.appendChild(valueDiv);
                    attrDiv.appendChild(itemDiv);
                });
            }
            
            eventDiv.appendChild(attrDiv);
        }
        
        eventDiv.onclick = () => {
            const span = currentTrace.find(s => s.span_id === event.span_id);
            if (span) showSpanDetails(span);
        };
        
        container.appendChild(eventDiv);
    });
}

function renderJSON() {
    const container = document.getElementById('json-viewer');
    container.innerHTML = '';
    
    if (!currentTrace) return;
    
    const pre = document.createElement('pre');
    pre.innerHTML = syntaxHighlightJSON(JSON.stringify(currentTrace, null, 2));
    container.appendChild(pre);
}

function syntaxHighlightJSON(json) {
    return json.replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g, function (match) {
        let cls = 'json-number';
        if (/^"/.test(match)) {
            if (/:$/.test(match)) {
                cls = 'json-key';
            } else {
                cls = 'json-string';
            }
        } else if (/true|false/.test(match)) {
            cls = 'json-boolean';
        } else if (/null/.test(match)) {
            cls = 'json-null';
        }
        return '<span class="' + cls + '">' + match + '</span>';
    });
}

function formatSpanTooltip(span) {
    let html = `<strong>${span.name}</strong><br>`;
    html += `Duration: ${span.duration_ms ? span.duration_ms.toFixed(2) + 'ms' : 'ongoing'}<br>`;
    html += `Status: ${span.status_code}<br>`;
    
    if (span.attributes.proxy_request_id) {
        html += `Request ID: ${span.attributes.proxy_request_id}<br>`;
    }
    
    if (span.events && span.events.length > 0) {
        html += `Events: ${span.events.length}<br>`;
    }
    
    return html;
}

function showSpanDetails(span) {
    const details = document.getElementById('span-details');
    const content = document.getElementById('details-content');
    
    content.innerHTML = '';
    
    // Basic info
    addDetailRow('Span ID', span.span_id);
    addDetailRow('Trace ID', span.trace_id);
    addDetailRow('Name', span.name);
    addDetailRow('Start Time', new Date(span.start_time * 1000).toLocaleString());
    if (span.end_time) {
        addDetailRow('End Time', new Date(span.end_time * 1000).toLocaleString());
    }
    if (span.duration_ms) {
        addDetailRow('Duration', span.duration_ms.toFixed(2) + ' ms');
    }
    addDetailRow('Status', span.status_code);
    if (span.status_message) {
        addDetailRow('Status Message', span.status_message);
    }
    
    // Attributes
    if (span.attributes && Object.keys(span.attributes).length > 0) {
        const attrHeader = document.createElement('div');
        attrHeader.className = 'detail-row';
        attrHeader.innerHTML = '<div class="detail-label">Attributes</div><div class="detail-value"></div>';
        content.appendChild(attrHeader);
        
        const attrSection = document.createElement('div');
        attrSection.className = 'attribute-section';
        
        Object.entries(span.attributes).forEach(([key, value]) => {
            const itemDiv = document.createElement('div');
            itemDiv.className = 'attribute-item';
            
            const keyDiv = document.createElement('div');
            keyDiv.className = 'attribute-key';
            keyDiv.textContent = key;
            itemDiv.appendChild(keyDiv);
            
            const valueDiv = document.createElement('div');
            valueDiv.className = 'attribute-value';
            
            // Format value based on type
            if (typeof value === 'object' && value !== null) {
                valueDiv.innerHTML = syntaxHighlightJSON(JSON.stringify(value, null, 2));
            } else if (typeof value === 'string' && value.length > 100) {
                valueDiv.textContent = value;
            } else {
                valueDiv.textContent = JSON.stringify(value);
            }
            
            itemDiv.appendChild(valueDiv);
            attrSection.appendChild(itemDiv);
        });
        
        content.appendChild(attrSection);
    }
    
    // Events
    if (span.events && span.events.length > 0) {
        addDetailRow('Events', '');
        
        // Check if this span has streaming chunks
        const streamingChunks = span.events.filter(e => e.name === 'streaming_chunk_received');
        
        if (streamingChunks.length > 0) {
            // Create a special streaming view
            const streamingSection = document.createElement('div');
            streamingSection.className = 'streaming-section';
            streamingSection.style.marginTop = '20px';
            
            const streamingHeader = document.createElement('h4');
            streamingHeader.textContent = `Streaming Chunks (${streamingChunks.length})`;
            streamingHeader.style.color = '#ff79c6';
            streamingSection.appendChild(streamingHeader);
            
            // Create chunks container
            const chunksContainer = document.createElement('div');
            chunksContainer.className = 'streaming-chunks';
            chunksContainer.style.maxHeight = '400px';
            chunksContainer.style.overflowY = 'auto';
            chunksContainer.style.backgroundColor = '#0d1117';
            chunksContainer.style.padding = '10px';
            chunksContainer.style.borderRadius = '6px';
            chunksContainer.style.fontFamily = 'monospace';
            chunksContainer.style.fontSize = '12px';
            
            streamingChunks.forEach((chunk, index) => {
                const chunkDiv = document.createElement('div');
                chunkDiv.style.marginBottom = '10px';
                chunkDiv.style.borderBottom = '1px solid #30363d';
                chunkDiv.style.paddingBottom = '10px';
                
                // Chunk header
                const chunkHeader = document.createElement('div');
                chunkHeader.style.color = '#8b949e';
                chunkHeader.style.marginBottom = '5px';
                chunkHeader.textContent = `Chunk ${chunk.attributes['proxy.streaming.chunk_index']} - ${new Date(chunk.timestamp * 1000).toLocaleTimeString()}`;
                chunkDiv.appendChild(chunkHeader);
                
                // Parse and display chunk data
                const chunkData = chunk.attributes['proxy.streaming.chunk_data'];
                if (chunkData) {
                    try {
                        // Try to parse as SSE data
                        if (chunkData.startsWith('data: ')) {
                            const jsonStr = chunkData.substring(6);
                            const parsed = JSON.parse(jsonStr);
                            
                            const chunkContent = document.createElement('pre');
                            chunkContent.style.margin = '0';
                            chunkContent.style.color = '#e6edf3';
                            chunkContent.innerHTML = syntaxHighlightJSON(JSON.stringify(parsed, null, 2));
                            chunkDiv.appendChild(chunkContent);
                        } else {
                            // Display raw data
                            const chunkContent = document.createElement('div');
                            chunkContent.style.color = '#e6edf3';
                            chunkContent.textContent = chunkData;
                            chunkDiv.appendChild(chunkContent);
                        }
                    } catch (e) {
                        // If parsing fails, show raw data
                        const chunkContent = document.createElement('div');
                        chunkContent.style.color = '#e6edf3';
                        chunkContent.textContent = chunkData;
                        chunkDiv.appendChild(chunkContent);
                    }
                }
                
                chunksContainer.appendChild(chunkDiv);
            });
            
            streamingSection.appendChild(chunksContainer);
            content.appendChild(streamingSection);
            
            // Show other events separately
            const otherEvents = span.events.filter(e => e.name !== 'streaming_chunk_received');
            if (otherEvents.length > 0) {
                addDetailRow('Other Events', '');
                otherEvents.forEach(event => {
                    addDetailRow(`  ${event.name}`, new Date(event.timestamp * 1000).toLocaleTimeString());
                });
            }
        } else {
            // Regular events display
            span.events.forEach(event => {
                addDetailRow(`  ${event.name}`, new Date(event.timestamp * 1000).toLocaleTimeString());
            });
        }
    }
    
    details.style.display = 'block';
    
    function addDetailRow(label, value) {
        const row = document.createElement('div');
        row.className = 'detail-row';
        
        const labelDiv = document.createElement('div');
        labelDiv.className = 'detail-label';
        labelDiv.textContent = label;
        
        const valueDiv = document.createElement('div');
        valueDiv.className = 'detail-value';
        valueDiv.textContent = value;
        
        row.appendChild(labelDiv);
        row.appendChild(valueDiv);
        content.appendChild(row);
    }
}

function clearVisualization() {
    document.getElementById('timeline').innerHTML = '';
    document.getElementById('events-list').innerHTML = '';
    document.getElementById('json-viewer').innerHTML = '';
    document.getElementById('span-details').style.display = 'none';
}

async function clearTraces() {
    if (confirm('Clear all stored traces?')) {
        await fetch('/telemetry/clear', { method: 'POST' });
        refreshTraces();
        clearVisualization();
    }
}

function exportTrace() {
    if (!currentTrace) {
        alert('No trace selected');
        return;
    }
    
    const blob = new Blob([JSON.stringify(currentTrace, null, 2)], {type: 'application/json'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `trace-${currentTrace[0].trace_id}.json`;
    a.click();
    URL.revokeObjectURL(url);
}

document.getElementById('trace-selector').addEventListener('change', (e) => {
    loadTrace(e.target.value);
});