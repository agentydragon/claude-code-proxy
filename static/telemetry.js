// Claude Code Proxy - Telemetry Viewer JavaScript

let autoRefreshInterval = null;
let allTraces = {};
let selectedTraceId = null;

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
        allTraces = data.traces;
        renderTraceList(data.traces);
    } catch (error) {
        console.error('Failed to refresh traces:', error);
    }
}

function updateStats(stats) {
    document.getElementById('total-traces').textContent = stats.total_traces;
    document.getElementById('total-spans').textContent = stats.total_spans;
    document.getElementById('avg-duration').textContent = Math.round(stats.avg_duration) + 'ms';
    document.getElementById('error-rate').textContent = Math.round(stats.error_rate) + '%';
}

function renderTraceList(traces) {
    const container = document.getElementById('trace-list');
    container.innerHTML = '';

    // Convert traces object to array and sort by start time (newest first)
    const traceArray = Object.entries(traces).map(([traceId, spans]) => {
        const rootSpan = spans.find(s => !s.parent_span_id) || spans[0];
        return { traceId, spans, rootSpan };
    }).sort((a, b) => b.rootSpan.start_time - a.rootSpan.start_time);

    traceArray.forEach(({ traceId, spans, rootSpan }) => {
        const traceDiv = document.createElement('div');
        traceDiv.className = 'trace-item';
        traceDiv.dataset.traceId = traceId;

        const isSelected = selectedTraceId === traceId;
        traceDiv.style.cssText = `
            padding: 12px;
            margin-bottom: 8px;
            background: ${isSelected ? '#1e3a5f' : '#1a1d24'};
            border-radius: 6px;
            cursor: pointer;
            transition: background 0.2s, border-color 0.2s;
            border: 2px solid ${isSelected ? '#60a5fa' : '#2a2d35'};
        `;

        // Determine status
        let status = 'OK';
        let statusColor = '#4ade80';

        if (!rootSpan.end_time) {
            status = 'ONGOING';
            statusColor = '#fbbf24';
        } else if (rootSpan.status_code === 'ERROR') {
            status = 'ERROR';
            statusColor = '#ef4444';
        } else if (rootSpan.duration_ms > 30000) {
            status = 'TIMEOUT';
            statusColor = '#ef4444';
        }

        // Extract prompt and response from events
        let promptText = 'No prompt found';
        let responseText = 'No response found';

        // Look for events in the root span only
        if (rootSpan.events) {
            rootSpan.events.forEach(event => {
                if (event.name === 'anthropic_request' && event.attributes['proxy.anthropic_request.body']) {
                    try {
                        const body = JSON.parse(event.attributes['proxy.anthropic_request.body']);
                        if (body.messages && body.messages.length > 0) {
                            const lastMessage = body.messages[body.messages.length - 1];
                            if (lastMessage.content) {
                                const content = typeof lastMessage.content === 'string'
                                    ? lastMessage.content
                                    : lastMessage.content.map(c => c.text || '').join(' ');
                                promptText = content.substring(0, 100) + (content.length > 100 ? '...' : '');
                            }
                        }
                    } catch (e) {
                        console.error('Failed to parse request body:', e);
                    }
                }

                if (event.name === 'anthropic_response' && event.attributes['proxy.anthropic_response.body']) {
                    try {
                        const body = JSON.parse(event.attributes['proxy.anthropic_response.body']);
                        if (body.content && body.content.length > 0) {
                            const textContent = body.content
                                .filter(c => c.type === 'text')
                                .map(c => c.text)
                                .join(' ');
                            if (textContent) {
                                responseText = textContent.substring(0, 100) + (textContent.length > 100 ? '...' : '');
                            }
                        }
                    } catch (e) {
                        console.error('Failed to parse response body:', e);
                    }
                }
            });
        }

        const timestamp = new Date(rootSpan.start_time * 1000).toLocaleTimeString();
        const duration = rootSpan.duration_ms ? `${Math.round(rootSpan.duration_ms)}ms` : '...';

        const promptClass = promptText === 'No prompt found' ? 'style="color: #6b7280;"' : 'style="color: #e5e7eb;"';
        const responseClass = responseText === 'No response found' ? 'style="color: #6b7280;"' : 'style="color: #e5e7eb;"';

        // Store request/response data on the div for easy access
        traceDiv.requestData = null;
        traceDiv.responseData = null;

        // Extract and store full request/response data
        if (rootSpan.events) {
            rootSpan.events.forEach(event => {
                if (event.name === 'anthropic_request' && event.attributes['proxy.anthropic_request.body']) {
                    try {
                        traceDiv.requestData = JSON.parse(event.attributes['proxy.anthropic_request.body']);
                    } catch (e) {}
                }
                if (event.name === 'anthropic_response' && event.attributes['proxy.anthropic_response.body']) {
                    try {
                        traceDiv.responseData = JSON.parse(event.attributes['proxy.anthropic_response.body']);
                    } catch (e) {}
                }
            });
        }

        traceDiv.innerHTML = `
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div style="display: flex; gap: 12px; align-items: center;">
                    <span style="color: ${statusColor}; font-weight: 600; min-width: 65px;">${status}</span>
                    <span style="color: #9ca3af;">${timestamp}</span>
                    <span style="color: #60a5fa;">${duration}</span>
                </div>
                <div style="display: flex; gap: 8px; align-items: center;">
                    ${traceDiv.requestData ? '<button class="json-btn" data-type="request" style="background: #1e3a5f; border: 1px solid #60a5fa; color: #60a5fa; padding: 2px 8px; border-radius: 4px; font-size: 0.8em; cursor: pointer;">Request JSON</button>' : ''}
                    ${traceDiv.responseData ? '<button class="json-btn" data-type="response" style="background: #1e3a5f; border: 1px solid #34d399; color: #34d399; padding: 2px 8px; border-radius: 4px; font-size: 0.8em; cursor: pointer;">Response JSON</button>' : ''}
                    <span style="color: #6b7280; font-size: 0.85em;">${traceId.substring(0, 8)}</span>
                </div>
            </div>
            <div style="margin-top: 8px; display: flex; gap: 20px; font-size: 0.95em;">
                <div style="flex: 1;">
                    <strong style="color: #a78bfa;">Prompt:</strong>
                    <span ${promptClass}>${escapeHtml(promptText)}</span>
                </div>
                <div style="flex: 1;">
                    <strong style="color: #34d399;">Response:</strong>
                    <span ${responseClass}>${escapeHtml(responseText)}</span>
                </div>
            </div>
        `;

        traceDiv.onmouseover = () => {
            if (selectedTraceId !== traceId) {
                traceDiv.style.background = '#22252d';
            }
        };

        traceDiv.onmouseout = () => {
            if (selectedTraceId !== traceId) {
                traceDiv.style.background = '#1a1d24';
            }
        };

        traceDiv.onclick = () => {
            // Update selection
            selectedTraceId = traceId;

            // Update all trace items to reflect new selection
            document.querySelectorAll('.trace-item').forEach(item => {
                const itemId = item.dataset.traceId;
                const isItemSelected = itemId === selectedTraceId;
                item.style.background = isItemSelected ? '#1e3a5f' : '#1a1d24';
                item.style.borderColor = isItemSelected ? '#60a5fa' : '#2a2d35';
            });

            showTraceDetails(traceId, spans);
        };

        container.appendChild(traceDiv);
    });
}

function escapeHtml(text) {
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return text.replace(/[&<>"']/g, m => map[m]);
}

function showTraceDetails(traceId, spans) {
    const details = document.getElementById('span-details');
    const content = document.getElementById('details-content');

    content.innerHTML = '';

    // Find root span
    const rootSpan = spans.find(s => !s.parent_span_id) || spans[0];

    // Add trace info
    const traceHeader = document.createElement('h4');
    traceHeader.textContent = 'Trace Information';
    traceHeader.style.color = '#e0e6ed';
    traceHeader.style.marginBottom = '15px';
    content.appendChild(traceHeader);

    addDetailRow('Trace ID', traceId);
    addDetailRow('Start Time', new Date(rootSpan.start_time * 1000).toLocaleString());
    if (rootSpan.end_time) {
        addDetailRow('End Time', new Date(rootSpan.end_time * 1000).toLocaleString());
    }
    if (rootSpan.duration_ms) {
        addDetailRow('Total Duration', Math.round(rootSpan.duration_ms) + ' ms');
    }
    addDetailRow('Status', rootSpan.status_code || 'OK');
    addDetailRow('Total Spans', spans.length.toString());

    // Add timeline of spans
    const timelineHeader = document.createElement('h4');
    timelineHeader.textContent = 'Span Timeline';
    timelineHeader.style.color = '#e0e6ed';
    timelineHeader.style.marginTop = '20px';
    timelineHeader.style.marginBottom = '15px';
    content.appendChild(timelineHeader);

    // Sort spans by start time
    const sortedSpans = [...spans].sort((a, b) => a.start_time - b.start_time);

    sortedSpans.forEach(span => {
        const spanDiv = document.createElement('div');
        spanDiv.style.cssText = `
            padding: 10px;
            margin-bottom: 10px;
            background: #0d1117;
            border-radius: 4px;
            border-left: 3px solid ${span.status_code === 'ERROR' ? '#ff5555' : '#5a7fdb'};
        `;

        const spanHeader = document.createElement('div');
        spanHeader.style.cssText = 'display: flex; justify-content: space-between; margin-bottom: 5px;';
        spanHeader.innerHTML = `
            <strong style="color: #64b5f6;">${span.name}</strong>
            <span style="color: #8892b0;">${span.duration_ms ? Math.round(span.duration_ms) + 'ms' : 'ongoing'}</span>
        `;
        spanDiv.appendChild(spanHeader);

        // Add key events for this span
        if (span.events && span.events.length > 0) {
            const eventList = document.createElement('div');
            eventList.style.cssText = 'margin-top: 5px; font-size: 0.9em;';

            span.events.forEach(event => {
                if (['anthropic_request', 'openai_request', 'anthropic_response', 'openai_response'].includes(event.name)) {
                    const eventItem = document.createElement('div');
                    eventItem.style.cssText = 'color: #8892b0; margin-top: 2px;';
                    eventItem.textContent = `• ${event.name}`;
                    eventList.appendChild(eventItem);
                }
            });

            spanDiv.appendChild(eventList);
        }

        spanDiv.onclick = () => showSpanDetails(span);
        spanDiv.style.cursor = 'pointer';

        content.appendChild(spanDiv);
    });

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
    html += `Duration: ${span.duration_ms ? Math.round(span.duration_ms) + 'ms' : 'ongoing'}<br>`;
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
        addDetailRow('Duration', Math.round(span.duration_ms) + ' ms');
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
        const eventsHeader = document.createElement('h4');
        eventsHeader.textContent = 'Events';
        eventsHeader.style.color = '#e0e6ed';
        eventsHeader.style.marginTop = '20px';
        eventsHeader.style.marginBottom = '10px';
        content.appendChild(eventsHeader);

        // Group events by type
        const eventGroups = {};
        span.events.forEach(event => {
            if (!eventGroups[event.name]) {
                eventGroups[event.name] = [];
            }
            eventGroups[event.name].push(event);
        });

        Object.entries(eventGroups).forEach(([eventName, events]) => {
            const eventRow = document.createElement('div');
            eventRow.style.cssText = `
                padding: 8px;
                margin-bottom: 5px;
                background: #0d1117;
                border-radius: 4px;
                cursor: pointer;
                transition: background 0.2s;
                user-select: none;
            `;

            // Add hover effect
            eventRow.onmouseenter = () => {
                if (!eventRow.dataset.expanded || eventRow.dataset.expanded === 'false') {
                    eventRow.style.background = '#161b22';
                }
            };
            eventRow.onmouseleave = () => {
                if (!eventRow.dataset.expanded || eventRow.dataset.expanded === 'false') {
                    eventRow.style.background = '#0d1117';
                }
            };

            const eventContent = document.createElement('div');
            eventContent.style.cssText = 'display: flex; justify-content: space-between; align-items: center;';

            const eventLabel = document.createElement('span');
            eventLabel.style.color = '#64b5f6';

            // Add expand indicator for expandable events
            const hasDetails = ['anthropic_request', 'openai_request', 'anthropic_response', 'openai_response', 'exception'].includes(eventName);
            if (hasDetails) {
                const arrow = document.createElement('span');
                arrow.style.cssText = 'display: inline-block; margin-right: 8px; transition: transform 0.2s;';
                arrow.textContent = '▶';
                arrow.className = 'expand-arrow';
                eventLabel.appendChild(arrow);
            }

            const nameSpan = document.createElement('span');
            nameSpan.textContent = eventName;
            eventLabel.appendChild(nameSpan);

            const eventCount = document.createElement('span');
            eventCount.style.color = '#8892b0';
            eventCount.textContent = events.length > 1 ? `(${events.length} events)` : '';

            eventContent.appendChild(eventLabel);
            eventContent.appendChild(eventCount);
            eventRow.appendChild(eventContent);

            // Add expandable details for events with meaningful data
            if (['anthropic_request', 'openai_request', 'anthropic_response', 'openai_response'].includes(eventName)) {
                const detailsDiv = document.createElement('div');
                detailsDiv.style.cssText = 'display: none; margin-top: 10px;';

                events.forEach(event => {
                    const eventDetail = document.createElement('div');
                    eventDetail.style.cssText = 'margin-top: 5px; padding: 5px; background: #161b22; border-radius: 3px;';

                    const timestamp = document.createElement('div');
                    timestamp.style.cssText = 'color: #8892b0; font-size: 0.85em; margin-bottom: 5px;';
                    timestamp.textContent = new Date(event.timestamp * 1000).toLocaleTimeString();
                    eventDetail.appendChild(timestamp);

                    // Show body preview for request/response events
                    const bodyAttr = event.attributes[`proxy.${eventName}.body`];
                    if (bodyAttr) {
                        try {
                            const body = JSON.parse(bodyAttr);
                            const preview = document.createElement('pre');
                            preview.style.cssText = 'margin: 0; font-size: 0.85em; max-height: 200px; overflow-y: auto;';
                            preview.innerHTML = syntaxHighlightJSON(JSON.stringify(body, null, 2));
                            eventDetail.appendChild(preview);
                        } catch (e) {
                            const preview = document.createElement('div');
                            preview.style.cssText = 'color: #8892b0; font-size: 0.85em;';
                            preview.textContent = 'Unable to parse body';
                            eventDetail.appendChild(preview);
                        }
                    }

                    detailsDiv.appendChild(eventDetail);
                });

                eventRow.appendChild(detailsDiv);

                eventRow.onclick = () => {
                    const isExpanded = eventRow.dataset.expanded === 'true';
                    eventRow.dataset.expanded = !isExpanded;
                    detailsDiv.style.display = isExpanded ? 'none' : 'block';
                    eventRow.style.background = isExpanded ? '#0d1117' : '#1a1d24';

                    const arrow = eventRow.querySelector('.expand-arrow');
                    if (arrow) {
                        arrow.style.transform = isExpanded ? 'rotate(0deg)' : 'rotate(90deg)';
                    }
                };
            } else if (eventName === 'streaming_chunk_received' && events.length > 0) {
                // For streaming chunks, just show summary
                const summaryDiv = document.createElement('div');
                summaryDiv.style.cssText = 'color: #8892b0; font-size: 0.85em; margin-top: 5px;';
                const firstChunk = events[0].attributes['proxy.streaming.chunk_index'] || 0;
                const lastChunk = events[events.length - 1].attributes['proxy.streaming.chunk_index'] || events.length - 1;
                summaryDiv.textContent = `Chunks ${firstChunk} - ${lastChunk}`;
                eventRow.appendChild(summaryDiv);
            } else if (eventName === 'exception') {
                // For exceptions, make them expandable to show stack trace
                const detailsDiv = document.createElement('div');
                detailsDiv.style.cssText = 'display: none; margin-top: 10px;';

                events.forEach(event => {
                    const eventDetail = document.createElement('div');
                    eventDetail.style.cssText = 'margin-top: 5px; padding: 10px; background: #161b22; border-radius: 3px; border-left: 3px solid #ef4444;';

                    // Look for exception details in attributes
                    const exceptionType = event.attributes['exception.type'] || 'Unknown Exception';
                    const exceptionMsg = event.attributes['exception.message'] || 'No message';
                    const stacktrace = event.attributes['exception.stacktrace'];

                    const exceptionHeader = document.createElement('div');
                    exceptionHeader.style.cssText = 'color: #ef4444; font-weight: 600; margin-bottom: 5px;';
                    exceptionHeader.textContent = exceptionType;
                    eventDetail.appendChild(exceptionHeader);

                    const exceptionMessage = document.createElement('div');
                    exceptionMessage.style.cssText = 'color: #fbbf24; margin-bottom: 10px;';
                    exceptionMessage.textContent = exceptionMsg;
                    eventDetail.appendChild(exceptionMessage);

                    if (stacktrace) {
                        const stackDiv = document.createElement('pre');
                        stackDiv.style.cssText = 'margin: 0; font-family: monospace; font-size: 0.85em; color: #e5e7eb; overflow-x: auto; white-space: pre-wrap;';
                        stackDiv.textContent = stacktrace;
                        eventDetail.appendChild(stackDiv);
                    }

                    detailsDiv.appendChild(eventDetail);
                });

                eventRow.appendChild(detailsDiv);

                // Make row red-tinted for exceptions
                eventRow.style.borderLeft = '3px solid #ef4444';

                eventRow.onclick = () => {
                    const isExpanded = eventRow.dataset.expanded === 'true';
                    eventRow.dataset.expanded = !isExpanded;
                    detailsDiv.style.display = isExpanded ? 'none' : 'block';
                    eventRow.style.background = isExpanded ? '#0d1117' : '#1a1d24';

                    const arrow = eventRow.querySelector('.expand-arrow');
                    if (arrow) {
                        arrow.style.transform = isExpanded ? 'rotate(0deg)' : 'rotate(90deg)';
                    }
                };
            }

            content.appendChild(eventRow);
        });
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
    document.getElementById('trace-list').innerHTML = '';
    document.getElementById('span-details').style.display = 'none';
}

async function clearTraces() {
    if (confirm('Clear all stored traces?')) {
        await fetch('/telemetry/clear', { method: 'POST' });
        refreshTraces();
        clearVisualization();
    }
}
