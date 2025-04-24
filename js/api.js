// API Service for handling all API requests
const ApiService = (function() {
    // Make HTTP requests
    async function makeRequest(url, method = 'GET', data = null, headers = {}) {
        const requestOptions = {
            method,
            headers: { ...CONFIG.DEFAULT_HEADERS, ...headers }
        };

        if (data) {
            if (data instanceof FormData) {
                // Don't set Content-Type for FormData, browser will set it automatically with boundary
                delete requestOptions.headers['Content-Type'];
                requestOptions.body = data;
            } else {
                requestOptions.body = JSON.stringify(data);
            }
        }

        try {
            const response = await fetch(CONFIG.API_BASE_URL + url, requestOptions);

            if (!response.ok) {
                const errorData = await response.json().catch(() => null);
                throw new Error(errorData?.error || `HTTP error! Status: ${response.status}`);
            }

            // Check content type
            const contentType = response.headers.get('content-type');
            if (contentType && contentType.includes('application/json')) {
                return await response.json();
            } else {
                return await response.text();
            }
        } catch (error) {
            logger.error('API request failed', error);
            throw error;
        }
    }

    // Authentication
    async function login(credentials) {
        return makeRequest('/auth/login', 'POST', credentials);
    }

    async function logout() {
        return makeRequest('/auth/logout');
    }

    // File Management
    async function uploadFile(formData) {
        return makeRequest('/api/v0/upload_spreadsheet_file', 'POST', formData);
    }

    async function deleteFile(fileName) {
        return makeRequest('/api/v0/delete_uploaded_file', 'POST', { fileName });
    }

    async function deleteAllFiles() {
        return makeRequest('/api/v0/delete_uploaded_files', 'POST');
    }

    // Question & SQL Generation
    async function generateQuestions() {
        return makeRequest('/api/v0/generate_questions');
    }

    async function generateSQL(question) {
        return makeRequest(`/api/v0/generate_sql?question=${encodeURIComponent(question)}`);
    }

    async function fixSQL(id, error) {
        return makeRequest('/api/v0/fix_sql', 'POST', { id, error });
    }

    async function updateSQL(id, sql) {
        return makeRequest('/api/v0/update_sql', 'POST', { id, sql });
    }

    // Data Handling
    async function runSQL(id) {
        return makeRequest(`/api/v0/run_sql?id=${encodeURIComponent(id)}`);
    }

    async function downloadCSV(id) {
        // For downloads, we'll use window.location to trigger the download
        window.location.href = `${CONFIG.API_BASE_URL}/api/v0/download_csv?id=${encodeURIComponent(id)}`;
        return true;
    }

    async function generatePlotly(id, chartInstructions = '') {
        let url = `/api/v0/generate_plotly_figure?id=${encodeURIComponent(id)}`;
        if (chartInstructions) {
            url += `&chart_instructions=${encodeURIComponent(chartInstructions)}`;
        }
        return makeRequest(url);
    }

    // Function Management
    async function createFunction(id) {
        return makeRequest(`/api/v0/create_function?id=${encodeURIComponent(id)}`);
    }

    async function updateFunction(oldFunctionName, updatedFunction) {
        return makeRequest('/api/v0/update_function', 'POST', { old_function_name: oldFunctionName, updated_function: updatedFunction });
    }

    async function deleteFunction(functionName) {
        return makeRequest('/api/v0/delete_function', 'POST', { function_name: functionName });
    }

    async function getAllFunctions() {
        return makeRequest('/api/v0/get_all_functions');
    }

    async function getFunction(question) {
        return makeRequest(`/api/v0/get_function?question=${encodeURIComponent(question)}`);
    }

    // Additional Features
    async function generateFollowupQuestions(id) {
        return makeRequest(`/api/v0/generate_followup_questions?id=${encodeURIComponent(id)}`);
    }

    async function generateSummary(id) {
        return makeRequest(`/api/v0/generate_summary?id=${encodeURIComponent(id)}`);
    }

    async function generateRewrittenQuestion(lastQuestion, newQuestion) {
        return makeRequest(`/api/v0/generate_rewritten_question?last_question=${encodeURIComponent(lastQuestion)}&new_question=${encodeURIComponent(newQuestion)}`);
    }

    // Question History
    async function getQuestionHistory() {
        return makeRequest('/api/v0/get_question_history');
    }

    async function loadQuestion(id) {
        return makeRequest(`/api/v0/load_question?id=${encodeURIComponent(id)}`);
    }

    // Configuration
    async function getConfig() {
        return makeRequest('/api/v0/get_config');
    }

    // Training Data
    async function getTrainingData() {
        return makeRequest('/api/v0/get_training_data');
    }

    async function addTrainingData(trainingData) {
        return makeRequest('/api/v0/train', 'POST', trainingData);
    }

    async function removeTrainingData(id) {
        return makeRequest('/api/v0/remove_training_data', 'POST', { id });
    }

    return {
        // Authentication
        login,
        logout,

        // File Management
        uploadFile,
        deleteFile,
        deleteAllFiles,

        // Question & SQL Generation
        generateQuestions,
        generateSQL,
        fixSQL,
        updateSQL,

        // Data Handling
        runSQL,
        downloadCSV,
        generatePlotly,

        // Function Management
        createFunction,
        updateFunction,
        deleteFunction,
        getAllFunctions,
        getFunction,

        // Additional Features
        generateFollowupQuestions,
        generateSummary,
        generateRewrittenQuestion,

        // Question History
        getQuestionHistory,
        loadQuestion,

        // Configuration
        getConfig,

        // Training Data
        getTrainingData,
        addTrainingData,
        removeTrainingData,

        // Utils
        makeRequest
    };
})();