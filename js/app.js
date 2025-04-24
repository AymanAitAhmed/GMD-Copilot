// Main Application
document.addEventListener('DOMContentLoaded', function() {
    // Initialize Config
    CONFIG.DEBUG = false; // Set to true to enable debug logging

    // Initialize Services
    AuthService.init();

    // Dashboard Service for managing the dashboard
    const DashboardService = {
        init: function() {
            // Initialize dashboard once user is authenticated
            if (AuthService.isLoggedIn()) {
                this.initializeDashboard();
            } else {
                AuthService.addAuthListener((user) => {
                    if (user) {
                        this.initializeDashboard();
                    }
                });
            }
        },

        initializeDashboard: function() {
            const dashboardContent = document.getElementById('dashboard-content');
            if (!dashboardContent) return;

            // Setup dashboard layout
            dashboardContent.innerHTML = `
                <div class="dashboard-grid">
                    <div class="sidebar">
                        <div id="file-manager-container"></div>
                    </div>
                    <div class="content">
                        <div id="sql-generator-container"></div>
                    </div>
                </div>
            `;

            // Initialize file manager
            FileManagerService.init().then(() => {
                const fileManagerContainer = document.getElementById('file-manager-container');
                if (fileManagerContainer) {
                    FileManagerService.render(fileManagerContainer);
                }
            });

            // Initialize SQL generator
            SQLGeneratorService.init().then(() => {
                const sqlGeneratorContainer = document.getElementById('sql-generator-container');
                if (sqlGeneratorContainer) {
                    SQLGeneratorService.render(sqlGeneratorContainer);
                }
            });

            // Add file listener to update when files change
            FileManagerService.addFileListener((files) => {
                // Update UI or show notification when files change
                if (files.length > 0) {
                    showToast(`${files.length} file(s) available for analysis`, 'info');
                }
            });

            logger.log('Dashboard initialized');
        }
    };

    // Make Dashboard Service globally available
    window.DashboardService = DashboardService;

    // Add some custom CSS to enhance the chat interface
    const customStyles = document.createElement('style');
    customStyles.textContent = `
        .chat-messages {
            max-height: 400px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: var(--space-4);
            padding: var(--space-4);
        }
        
        .chat-message {
            animation: fadeIn 0.3s ease;
        }
        
        .chat-message-user, .chat-message-system {
            display: flex;
            flex-direction: column;
            max-width: 80%;
        }
        
        .chat-message-user {
            align-items: flex-end;
            align-self: flex-end;
        }
        
        .chat-message-system {
            align-items: flex-start;
            align-self: flex-start;
        }
        
        .chat-bubble {
            padding: var(--space-3) var(--space-4);
            border-radius: var(--radius-lg);
            margin-bottom: var(--space-1);
        }
        
        .chat-bubble-user {
            background-color: var(--color-primary);
            color: white;
            border-radius: var(--radius-lg) var(--radius-lg) 0 var(--radius-lg);
        }
        
        .chat-bubble-system {
            background-color: var(--color-gray-100);
            color: var(--color-gray-800);
            border-radius: var(--radius-lg) var(--radius-lg) var(--radius-lg) 0;
        }
        
        .chat-message-info {
            display: flex;
            font-size: 0.75rem;
            color: var(--color-gray-500);
            gap: var(--space-2);
        }
        
        .suggested-questions-header {
            margin-bottom: var(--space-2);
            font-weight: 500;
            color: var(--color-gray-700);
        }
        
        .suggested-questions-list {
            display: flex;
            flex-wrap: wrap;
            gap: var(--space-2);
        }
        
        .file-list-empty {
            text-align: center;
            padding: var(--space-4);
            color: var(--color-gray-500);
            font-style: italic;
        }
        
        #results-container {
            margin-top: var(--space-6);
        }
        
        .pagination {
            display: flex;
            justify-content: center;
            align-items: center;
            margin-top: var(--space-4);
            gap: var(--space-2);
        }
        
        .pagination-buttons {
            display: flex;
            gap: var(--space-1);
        }
        
        .pagination-button {
            display: flex;
            align-items: center;
            justify-content: center;
            min-width: 32px;
            height: 32px;
            border-radius: var(--radius);
            border: 1px solid var(--color-gray-300);
            background-color: white;
            color: var(--color-gray-700);
            cursor: pointer;
            transition: all 0.2s ease;
        }
        
        .pagination-button:hover {
            background-color: var(--color-gray-100);
        }
        
        .pagination-button.active {
            background-color: var(--color-primary);
            color: white;
            border-color: var(--color-primary);
        }
        
        .pagination-button:disabled {
            background-color: var(--color-gray-100);
            color: var(--color-gray-400);
            cursor: not-allowed;
        }
    `;

    document.head.appendChild(customStyles);
});