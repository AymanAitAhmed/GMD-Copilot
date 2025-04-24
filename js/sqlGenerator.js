// SQL Generator Service
const SQLGeneratorService = (function() {
    let currentQuestion = '';
    let currentSQL = '';
    let currentId = '';
    let currentResults = null;
    let currentChart = null;
    let followupQuestions = [];
    let summary = '';

    // Initialize SQL Generator
    function init() {
        return Promise.resolve();
    }

    // Generate SQL from a question
    async function generateSQL(question) {
        if (!question) {
            return { success: false, message: 'No question provided' };
        }

        currentQuestion = question;

        try {
            setLoading('#chat-container', true);

            const response = await ApiService.generateSQL(question);

            if (response && (response.type === 'sql' || response.type === 'text')) {
                currentId = response.id;
                currentSQL = response.text;

                return {
                    success: true,
                    id: response.id,
                    sql: response.text,
                    isValidSQL: response.type === 'sql'
                };
            } else {
                throw new Error(response?.error || 'Failed to generate SQL');
            }
        } catch (error) {
            logger.error('Generate SQL error', error);
            return { success: false, message: error.message || 'Failed to generate SQL' };
        } finally {
            setLoading('#chat-container', false);
        }
    }

    // Run SQL query
    async function runSQL(id, sql) {
        if (!id) {
            return { success: false, message: 'No ID provided' };
        }

        try {
            setLoading('#results-container', true);

            // If SQL was modified, update it first
            if (sql && sql !== currentSQL) {
                await ApiService.updateSQL(id, sql);
                currentSQL = sql;
            }

            const response = await ApiService.runSQL(id);

            if (response) {
                if (response.type === 'df') {
                    currentResults = JSON.parse(response.df);

                    return {
                        success: true,
                        results: currentResults,
                        shouldGenerateChart: response.should_generate_chart
                    };
                } else if (response.type === 'sql_error') {
                    return { success: false, message: response.error, isSQLError: true };
                } else {
                    throw new Error(response?.error || 'Failed to run SQL');
                }
            } else {
                throw new Error('Failed to run SQL');
            }
        } catch (error) {
            logger.error('Run SQL error', error);
            return { success: false, message: error.message || 'Failed to run SQL' };
        } finally {
            setLoading('#results-container', false);
        }
    }

    // Fix SQL error
    async function fixSQL(id, error) {
        if (!id || !error) {
            return { success: false, message: 'Missing ID or error message' };
        }

        try {
            setLoading('#sql-editor', true);

            const response = await ApiService.fixSQL(id, error);

            if (response && response.type === 'sql') {
                currentSQL = response.text;

                return { success: true, sql: response.text };
            } else {
                throw new Error(response?.error || 'Failed to fix SQL');
            }
        } catch (error) {
            logger.error('Fix SQL error', error);
            return { success: false, message: error.message || 'Failed to fix SQL' };
        } finally {
            setLoading('#sql-editor', false);
        }
    }

    // Generate chart
    async function generateChart(id, chartInstructions = '') {
        if (!id) {
            return { success: false, message: 'No ID provided' };
        }

        try {
            setLoading('#chart-container', true);

            const response = await ApiService.generatePlotly(id, chartInstructions);

            if (response && response.type === 'plotly_figure') {
                currentChart = JSON.parse(response.fig);

                return { success: true, chart: currentChart };
            } else {
                throw new Error(response?.error || 'Failed to generate chart');
            }
        } catch (error) {
            logger.error('Generate chart error', error);
            return { success: false, message: error.message || 'Failed to generate chart' };
        } finally {
            setLoading('#chart-container', false);
        }
    }

    // Generate followup questions
    async function generateFollowupQuestions(id) {
        if (!id) {
            return { success: false, message: 'No ID provided' };
        }

        try {
            const response = await ApiService.generateFollowupQuestions(id);

            if (response && response.type === 'question_list') {
                followupQuestions = response.questions || [];

                return {
                    success: true,
                    questions: followupQuestions,
                    header: response.header
                };
            } else {
                throw new Error(response?.error || 'Failed to generate followup questions');
            }
        } catch (error) {
            logger.error('Generate followup questions error', error);
            return { success: false, message: error.message || 'Failed to generate followup questions' };
        }
    }

    // Generate summary
    async function generateSummary(id) {
        if (!id) {
            return { success: false, message: 'No ID provided' };
        }

        try {
            setLoading('#summary-container', true);

            const response = await ApiService.generateSummary(id);

            if (response && (response.type === 'html' || response.type === 'text')) {
                summary = response.text;

                return { success: true, summary: summary };
            } else {
                throw new Error(response?.error || 'Failed to generate summary');
            }
        } catch (error) {
            logger.error('Generate summary error', error);
            return { success: false, message: error.message || 'Failed to generate summary' };
        } finally {
            setLoading('#summary-container', false);
        }
    }

    // Get suggested questions
    async function getSuggestedQuestions() {
        try {
            const response = await ApiService.generateQuestions();

            if (response && response.type === 'question_list') {
                return {
                    success: true,
                    questions: response.questions || [],
                    header: response.header
                };
            } else {
                throw new Error(response?.error || 'Failed to get suggested questions');
            }
        } catch (error) {
            logger.error('Get suggested questions error', error);
            return { success: false, message: error.message || 'Failed to get suggested questions' };
        }
    }

    // Download results as CSV
    function downloadCSV(id) {
        if (!id) {
            showToast('Cannot download CSV: Missing ID', 'error');
            return false;
        }

        try {
            ApiService.downloadCSV(id);
            return true;
        } catch (error) {
            logger.error('Download CSV error', error);
            showToast('Failed to download CSV', 'error');
            return false;
        }
    }

    // Create a function from current query
    async function createFunction(id) {
        if (!id) {
            return { success: false, message: 'No ID provided' };
        }

        try {
            setLoading('#function-button', true);

            const response = await ApiService.createFunction(id);

            if (response && response.type === 'function_template') {
                return {
                    success: true,
                    functionTemplate: response.function_template
                };
            } else {
                throw new Error(response?.error || 'Failed to create function');
            }
        } catch (error) {
            logger.error('Create function error', error);
            return { success: false, message: error.message || 'Failed to create function' };
        } finally {
            setLoading('#function-button', false);
        }
    }

    // Render SQL generator UI
    function render(container) {
        if (!container) return;

        container.innerHTML = `
            <div class="card" id="chat-container">
                <div class="card-header">
                    <h3 class="card-title">Data Insights Chatbot</h3>
                    <p class="card-subtitle">Ask questions about your data</p>
                </div>
                <div class="card-body">
                    <div id="chat-messages" class="chat-messages"></div>
                    
                    <div id="suggested-questions-container" class="suggested-questions">
                        <!-- Suggested questions will be loaded here -->
                    </div>
                </div>
                <div class="card-footer">
                    <div class="question-input">
                        <textarea id="question-input" class="form-control" 
                            placeholder="Ask a question about your data..." rows="2"></textarea>
                        <button id="ask-button" class="btn btn-primary">
                            <i class="fas fa-paper-plane"></i>
                        </button>
                    </div>
                </div>
            </div>
            
            <div id="results-container" class="results-container hidden">
                <!-- Results will be displayed here -->
            </div>
        `;

        // Add event listeners
        setupChatInteraction(container);

        // Load suggested questions
        loadSuggestedQuestions(container);
    }

    // Setup chat interaction
    function setupChatInteraction(container) {
        const questionInput = container.querySelector('#question-input');
        const askButton = container.querySelector('#ask-button');
        const chatMessages = container.querySelector('#chat-messages');

        if (questionInput && askButton) {
            // Send question on button click
            askButton.addEventListener('click', () => {
                const question = questionInput.value.trim();
                if (question) {
                    sendQuestion(question, chatMessages);
                    questionInput.value = '';
                }
            });

            // Send question on Enter key (but allow Shift+Enter for new line)
            questionInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    const question = questionInput.value.trim();
                    if (question) {
                        sendQuestion(question, chatMessages);
                        questionInput.value = '';
                    }
                }
            });
        }
    }

    // Load suggested questions
    async function loadSuggestedQuestions(container) {
        const suggestedQuestionsContainer = container.querySelector('#suggested-questions-container');
        if (!suggestedQuestionsContainer) return;

        const result = await getSuggestedQuestions();

        if (result.success && result.questions && result.questions.length > 0) {
            suggestedQuestionsContainer.innerHTML = `
                <div class="suggested-questions-header">${result.header || 'Suggested questions:'}</div>
                <div class="suggested-questions-list">
                    ${result.questions.map(question => `
                        <div class="suggested-question" data-question="${question}">${question}</div>
                    `).join('')}
                </div>
            `;

            // Add event listeners to suggested questions
            const suggestedQuestionElements = suggestedQuestionsContainer.querySelectorAll('.suggested-question');
            const chatMessages = container.querySelector('#chat-messages');

            suggestedQuestionElements.forEach(element => {
                element.addEventListener('click', () => {
                    const question = element.dataset.question;
                    if (question) {
                        sendQuestion(question, chatMessages);
                    }
                });
            });
        } else {
            suggestedQuestionsContainer.innerHTML = '';
        }
    }

    // Send a question and process the response
    async function sendQuestion(question, chatMessages) {
        if (!chatMessages) return;

        // Add user question to chat
        appendMessage(chatMessages, 'user', question);

        // Generate SQL
        const generateResult = await generateSQL(question);

        if (generateResult.success) {
            // Add system response to chat
            const responseMessage = `I've generated SQL for your question. ${generateResult.isValidSQL ? "Let's run it!" : "There seems to be an issue with the SQL."}`;
            appendMessage(chatMessages, 'system', responseMessage);

            // Show SQL and run it if valid
            const resultsContainer = document.getElementById('results-container');
            if (resultsContainer) {
                resultsContainer.classList.remove('hidden');

                renderSQLResults(resultsContainer, generateResult.id, generateResult.sql, generateResult.isValidSQL);

                if (generateResult.isValidSQL) {
                    // Run the SQL
                    await runSQLAndShowResults(generateResult.id, generateResult.sql, resultsContainer);
                }
            }
        } else {
            // Add error message to chat
            appendMessage(chatMessages, 'system', `Sorry, I couldn't generate SQL for your question: ${generateResult.message}`);
        }

        // Scroll to the bottom of the chat
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Append a message to the chat
    function appendMessage(chatMessages, sender, message) {
        const messageElement = document.createElement('div');
        messageElement.className = 'chat-message';

        const timestamp = new Date().toLocaleTimeString();

        if (sender === 'user') {
            messageElement.innerHTML = `
                <div class="chat-message-user">
                    <div class="chat-bubble chat-bubble-user">${message}</div>
                    <div class="chat-message-info">
                        <span class="chat-message-time">${timestamp}</span>
                        <span class="chat-message-name">You</span>
                    </div>
                </div>
            `;
        } else {
            messageElement.innerHTML = `
                <div class="chat-message-system">
                    <div class="chat-bubble chat-bubble-system">${message}</div>
                    <div class="chat-message-info">
                        <span class="chat-message-name">Chatbot</span>
                        <span class="chat-message-time">${timestamp}</span>
                    </div>
                </div>
            `;
        }

        chatMessages.appendChild(messageElement);
    }

    // Render SQL results
    function renderSQLResults(container, id, sql, isValidSQL) {
        container.innerHTML = `
            <div class="card" id="sql-container">
                <div class="card-header">
                    <h3 class="card-title">SQL Query</h3>
                    <div class="card-actions">
                        <button id="edit-sql-button" class="btn btn-sm btn-outline">
                            <i class="fas fa-edit btn-icon"></i>
                            Edit
                        </button>
                    </div>
                </div>
                <div class="card-body">
                    <div id="sql-editor" class="sql-display">${sql}</div>
                    
                    <div class="sql-actions">
                        ${isValidSQL ? `
                            <button id="run-sql-button" class="btn btn-primary">
                                <i class="fas fa-play btn-icon"></i>
                                Run Query
                            </button>
                        ` : `
                            <button id="fix-sql-button" class="btn btn-warning">
                                <i class="fas fa-wrench btn-icon"></i>
                                Fix SQL
                            </button>
                        `}
                    </div>
                </div>
            </div>
            
            <div id="data-results-container" class="hidden">
                <!-- Data results will be displayed here -->
            </div>
        `;

        // Add event listeners
        const editSqlButton = container.querySelector('#edit-sql-button');
        const sqlEditor = container.querySelector('#sql-editor');
        const runSqlButton = container.querySelector('#run-sql-button');
        const fixSqlButton = container.querySelector('#fix-sql-button');

        if (editSqlButton && sqlEditor) {
            editSqlButton.addEventListener('click', () => {
                const isEditing = sqlEditor.getAttribute('contenteditable') === 'true';

                if (isEditing) {
                    // Save changes
                    sqlEditor.setAttribute('contenteditable', 'false');
                    editSqlButton.innerHTML = '<i class="fas fa-edit btn-icon"></i> Edit';
                    currentSQL = sqlEditor.textContent;
                } else {
                    // Enable editing
                    sqlEditor.setAttribute('contenteditable', 'true');
                    sqlEditor.focus();
                    editSqlButton.innerHTML = '<i class="fas fa-save btn-icon"></i> Save';
                }
            });
        }

        if (runSqlButton) {
            runSqlButton.addEventListener('click', async () => {
                await runSQLAndShowResults(id, sqlEditor.textContent, container);
            });
        }

        if (fixSqlButton) {
            fixSqlButton.addEventListener('click', async () => {
                const error = 'The SQL query has syntax or semantic errors.';
                const result = await fixSQL(id, error);

                if (result.success) {
                    sqlEditor.textContent = result.sql;

                    // Replace fix button with run button
                    const sqlActions = container.querySelector('.sql-actions');
                    if (sqlActions) {
                        sqlActions.innerHTML = `
                            <button id="run-sql-button" class="btn btn-primary">
                                <i class="fas fa-play btn-icon"></i>
                                Run Query
                            </button>
                        `;

                        const newRunSqlButton = sqlActions.querySelector('#run-sql-button');
                        if (newRunSqlButton) {
                            newRunSqlButton.addEventListener('click', async () => {
                                await runSQLAndShowResults(id, sqlEditor.textContent, container);
                            });
                        }
                    }

                    showToast('SQL query has been fixed', 'success');
                } else {
                    showToast(`Failed to fix SQL: ${result.message}`, 'error');
                }
            });
        }
    }

    // Run SQL and show results
    async function runSQLAndShowResults(id, sql, container) {
        const dataResultsContainer = container.querySelector('#data-results-container');
        if (!dataResultsContainer) return;

        const result = await runSQL(id, sql);

        if (result.success) {
            dataResultsContainer.classList.remove('hidden');

            // Render data table
            renderDataTable(dataResultsContainer, id, result.results);

            // Generate chart if appropriate
            if (result.shouldGenerateChart) {
                await generateAndRenderChart(dataResultsContainer, id);
            }

            // Generate followup questions
            await generateAndRenderFollowupQuestions(dataResultsContainer, id);

            // Generate summary
            await generateAndRenderSummary(dataResultsContainer, id);
        } else {
            dataResultsContainer.classList.add('hidden');

            if (result.isSQLError) {
                showToast(`SQL Error: ${result.message}`, 'error');

                // Show fix button
                const sqlActions = container.querySelector('.sql-actions');
                if (sqlActions) {
                    sqlActions.innerHTML = `
                        <button id="fix-sql-button" class="btn btn-warning">
                            <i class="fas fa-wrench btn-icon"></i>
                            Fix SQL
                        </button>
                    `;

                    const fixSqlButton = sqlActions.querySelector('#fix-sql-button');
                    if (fixSqlButton) {
                        fixSqlButton.addEventListener('click', async () => {
                            const fixResult = await fixSQL(id, result.message);

                            if (fixResult.success) {
                                const sqlEditor = container.querySelector('#sql-editor');
                                if (sqlEditor) {
                                    sqlEditor.textContent = fixResult.sql;
                                }

                                // Replace fix button with run button
                                sqlActions.innerHTML = `
                                    <button id="run-sql-button" class="btn btn-primary">
                                        <i class="fas fa-play btn-icon"></i>
                                        Run Query
                                    </button>
                                `;

                                const newRunSqlButton = sqlActions.querySelector('#run-sql-button');
                                if (newRunSqlButton) {
                                    newRunSqlButton.addEventListener('click', async () => {
                                        await runSQLAndShowResults(id, sqlEditor.textContent, container);
                                    });
                                }

                                showToast('SQL query has been fixed', 'success');
                            } else {
                                showToast(`Failed to fix SQL: ${fixResult.message}`, 'error');
                            }
                        });
                    }
                }
            } else {
                showToast(`Error: ${result.message}`, 'error');
            }
        }
    }

    // Render data table
    function renderDataTable(container, id, results) {
        const resultsSection = document.createElement('div');
        resultsSection.className = 'section';
        resultsSection.innerHTML = `
            <div class="section-header">
                <div class="results-header">
                    <h3 class="results-title">Query Results</h3>
                    <div class="results-actions">
                        <button id="download-csv-button" class="btn btn-outline">
                            <i class="fas fa-download btn-icon"></i>
                            Download CSV
                        </button>
                        <button id="function-button" class="btn btn-accent">
                            <i class="fas fa-save btn-icon"></i>
                            Save as Function
                        </button>
                    </div>
                </div>
            </div>
            <div class="table-container">
                <table class="table table-striped">
                    <thead>
                        <tr>
                            ${results.length > 0 ? Object.keys(results[0]).map(key => `
                                <th>${key}</th>
                            `).join('') : '<th>No data</th>'}
                        </tr>
                    </thead>
                    <tbody>
                        ${results.length > 0 ? results.map(row => `
                            <tr>
                                ${Object.values(row).map(value => `
                                    <td>${value !== null ? value : 'null'}</td>
                                `).join('')}
                            </tr>
                        `).join('') : '<tr><td>No results found</td></tr>'}
                    </tbody>
                </table>
            </div>
        `;

        // Add to container
        container.innerHTML = '';
        container.appendChild(resultsSection);

        // Add event listeners
        const downloadCsvButton = container.querySelector('#download-csv-button');
        const functionButton = container.querySelector('#function-button');

        if (downloadCsvButton) {
            downloadCsvButton.addEventListener('click', () => {
                downloadCSV(id);
            });
        }

        if (functionButton) {
            functionButton.addEventListener('click', async () => {
                const result = await createFunction(id);

                if (result.success) {
                    // Show function template in a modal
                    showFunctionModal(result.functionTemplate);
                } else {
                    showToast(`Failed to create function: ${result.message}`, 'error');
                }
            });
        }
    }

    // Generate and render chart
    async function generateAndRenderChart(container, id) {
        const result = await generateChart(id);

        if (result.success) {
            const chartSection = document.createElement('div');
            chartSection.className = 'section';
            chartSection.innerHTML = `
                <div class="section-header">
                    <div class="results-header">
                        <h3 class="results-title">Data Visualization</h3>
                        <div class="results-actions">
                            <button id="redraw-chart-button" class="btn btn-outline">
                                <i class="fas fa-sync-alt btn-icon"></i>
                                Redraw Chart
                            </button>
                        </div>
                    </div>
                </div>
                <div id="chart-container" class="chart-container"></div>
            `;

            // Add to container
            container.appendChild(chartSection);

            // Render chart
            const chartContainer = chartSection.querySelector('#chart-container');
            if (chartContainer) {
                Plotly.newPlot(chartContainer, result.chart.data, result.chart.layout);
            }

            // Add event listeners
            const redrawChartButton = chartSection.querySelector('#redraw-chart-button');
            if (redrawChartButton) {
                redrawChartButton.addEventListener('click', () => {
                    // Show modal to input chart instructions
                    showChartInstructionsModal(id, chartContainer);
                });
            }
        }
    }

    // Generate and render followup questions
    async function generateAndRenderFollowupQuestions(container, id) {
        const result = await generateFollowupQuestions(id);

        if (result.success && result.questions && result.questions.length > 0) {
            const followupSection = document.createElement('div');
            followupSection.className = 'section';
            followupSection.innerHTML = `
                <div class="section-header">
                    <h3 class="results-title">${result.header || 'Follow-up Questions'}</h3>
                </div>
                <div class="suggested-questions">
                    ${result.questions.map(question => `
                        <div class="suggested-question" data-question="${question}">${question}</div>
                    `).join('')}
                </div>
            `;

            // Add to container
            container.appendChild(followupSection);

            // Add event listeners
            const suggestedQuestions = followupSection.querySelectorAll('.suggested-question');
            suggestedQuestions.forEach(element => {
                element.addEventListener('click', () => {
                    const question = element.dataset.question;
                    if (question) {
                        const chatMessages = document.getElementById('chat-messages');
                        if (chatMessages) {
                            sendQuestion(question, chatMessages);
                        }
                    }
                });
            });
        }
    }

    // Generate and render summary
    async function generateAndRenderSummary(container, id) {
        const result = await generateSummary(id);

        if (result.success) {
            const summarySection = document.createElement('div');
            summarySection.className = 'section';
            summarySection.innerHTML = `
                <div class="section-header">
                    <h3 class="results-title">Summary</h3>
                </div>
                <div id="summary-container" class="card">
                    <div class="card-body">
                        ${result.summary}
                    </div>
                </div>
            `;

            // Add to container
            container.appendChild(summarySection);
        }
    }

    // Show chart instructions modal
    function showChartInstructionsModal(id, chartContainer) {
        const modalContent = createElement('div', {}, [
            createElement('div', { className: 'form-group' }, [
                createElement('label', {
                    className: 'form-label',
                    htmlFor: 'chart-instructions',
                    textContent: 'Chart Instructions'
                }),
                createElement('textarea', {
                    className: 'form-control',
                    id: 'chart-instructions',
                    placeholder: 'Enter specific instructions for the chart (e.g., "Use a bar chart", "Show data by month", etc.)',
                    rows: 4
                })
            ])
        ]);

        showModal('Redraw Chart', modalContent, null, [
            {
                text: 'Cancel',
                type: 'outline',
                onClick: (closeModal) => closeModal()
            },
            {
                text: 'Generate Chart',
                type: 'primary',
                onClick: async (closeModal) => {
                    const instructionsInput = document.getElementById('chart-instructions');
                    if (instructionsInput) {
                        const chartInstructions = instructionsInput.value.trim();

                        closeModal();

                        // Generate new chart
                        const result = await generateChart(id, chartInstructions);

                        if (result.success && chartContainer) {
                            // Update chart
                            Plotly.react(chartContainer, result.chart.data, result.chart.layout);
                            showToast('Chart has been updated', 'success');
                        } else {
                            showToast(`Failed to update chart: ${result.message}`, 'error');
                        }
                    }
                }
            }
        ]);
    }

    // Show function modal
    function showFunctionModal(functionTemplate) {
        const functionJson = JSON.stringify(functionTemplate, null, 2);

        const modalContent = createElement('div', {}, [
            createElement('p', {
                textContent: 'Your function has been created. You can use this function in your application.'
            }),
            createElement('div', {
                className: 'sql-display',
                style: { marginTop: 'var(--space-4)' },
                textContent: functionJson
            }),
            createElement('div', {
                className: 'form-text',
                textContent: 'Copy this function to use it in your application or API.'
            })
        ]);

        showModal('Function Created', modalContent, null, [
            {
                text: 'Close',
                type: 'outline',
                onClick: (closeModal) => closeModal()
            },
            {
                text: 'Copy to Clipboard',
                type: 'primary',
                onClick: (closeModal) => {
                    navigator.clipboard.writeText(functionJson)
                        .then(() => {
                            showToast('Function copied to clipboard', 'success');
                        })
                        .catch((error) => {
                            logger.error('Failed to copy function', error);
                            showToast('Failed to copy function', 'error');
                        });
                }
            }
        ]);
    }

    return {
        init,
        generateSQL,
        runSQL,
        fixSQL,
        generateChart,
        generateFollowupQuestions,
        generateSummary,
        getSuggestedQuestions,
        downloadCSV,
        createFunction,
        render,
        getCurrentQuestion: () => currentQuestion,
        getCurrentSQL: () => currentSQL,
        getCurrentId: () => currentId,
        getCurrentResults: () => currentResults,
        getCurrentChart: () => currentChart,
        getFollowupQuestions: () => followupQuestions,
        getSummary: () => summary
    };
})();