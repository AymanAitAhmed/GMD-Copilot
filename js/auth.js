// Authentication Service
const AuthService = (function() {
    let currentUser = null;
    let authListeners = [];

    // Initialize auth state
    function init() {
        // Check if user is logged in
        const storedUser = localStorage.getItem('user');
        if (storedUser) {
            try {
                currentUser = JSON.parse(storedUser);
                notifyListeners();
            } catch (e) {
                logger.error('Failed to parse stored user', e);
                localStorage.removeItem('user');
            }
        }

        // Render auth UI
        renderAuthUI();
    }

    // Render the authentication UI
    function renderAuthUI() {
        const authContainer = document.getElementById('auth-container');
        const dashboardContainer = document.getElementById('dashboard-container');

        if (!authContainer || !dashboardContainer) {
            logger.error('Auth or dashboard container not found');
            return;
        }

        if (isLoggedIn()) {
            // User is logged in, hide auth container and show dashboard
            hideElement(authContainer);
            showElement(dashboardContainer);

            // Render user info in the dashboard
            renderUserInfo();

            // Initialize dashboard
            if (typeof DashboardService !== 'undefined') {
                DashboardService.init();
            }
        } else {
            // User is not logged in, show auth container and hide dashboard
            showElement(authContainer);
            hideElement(dashboardContainer);

            // Render login form
            renderLoginForm();
        }
    }

    // Render login form
    function renderLoginForm() {
        const authContainer = document.getElementById('auth-container');
        if (!authContainer) return;

        authContainer.innerHTML = `
            <div class="card login-card">
                <div class="card-header">
                    <h2 class="card-title">Data Insights Chatbot</h2>
                    <p class="card-subtitle">Sign in to continue</p>
                </div>
                <div class="card-body">
                    <form id="login-form">
                        <div class="form-group">
                            <label for="email" class="form-label">Email</label>
                            <input type="email" id="email" class="form-control" required>
                        </div>
                        <div class="form-group">
                            <label for="password" class="form-label">Password</label>
                            <input type="password" id="password" class="form-control" required>
                        </div>
                        <div id="login-error" class="form-error hidden"></div>
                        <button type="submit" class="btn btn-primary" style="width: 100%; margin-top: var(--space-4);">
                            Sign In
                        </button>
                    </form>
                </div>
                <div class="card-footer" style="text-align: center;">
                    <p>Don't have an account? <a href="#" id="register-link">Sign up</a></p>
                </div>
            </div>
        `;

        // Add event listeners
        const loginForm = document.getElementById('login-form');
        if (loginForm) {
            loginForm.addEventListener('submit', async (e) => {
                e.preventDefault();

                const emailInput = document.getElementById('email');
                const passwordInput = document.getElementById('password');
                const loginErrorElement = document.getElementById('login-error');

                if (!emailInput || !passwordInput || !loginErrorElement) return;

                const email = emailInput.value.trim();
                const password = passwordInput.value;

                if (!email || !password) {
                    loginErrorElement.textContent = 'Please enter both email and password';
                    showElement(loginErrorElement);
                    return;
                }

                try {
                    setLoading(loginForm, true);
                    hideElement(loginErrorElement);

                    const response = await ApiService.login({ email, password });

                    if (response && response.type === 'not_logged_in') {
                        // If the response contains HTML for a custom login form
                        if (response.html) {
                            authContainer.innerHTML = response.html;

                            // Process any scripts in the HTML
                            const scripts = authContainer.getElementsByTagName('script');
                            for (let i = 0; i < scripts.length; i++) {
                                eval(scripts[i].innerText);
                            }
                        } else {
                            throw new Error('Authentication failed');
                        }
                    } else {
                        // For demo purposes, we'll mock a successful login since we don't have a real backend
                        setUser({
                            id: '1',
                            name: email.split('@')[0],
                            email: email,
                            role: 'user'
                        });
                    }
                } catch (error) {
                    loginErrorElement.textContent = error.message || 'Authentication failed';
                    showElement(loginErrorElement);
                } finally {
                    setLoading(loginForm, false);
                }
            });
        }

        // For demo purposes, we'll add a mock login button
        const registerLink = document.getElementById('register-link');
        if (registerLink) {
            registerLink.addEventListener('click', (e) => {
                e.preventDefault();

                // For demo purposes, we'll just show a message
                showToast('Registration is not available in this demo. Please use mock login.', 'info');

                // Add a mock login button
                const loginCard = document.querySelector('.login-card');
                if (loginCard) {
                    const mockLoginButton = document.createElement('button');
                    mockLoginButton.className = 'btn btn-secondary';
                    mockLoginButton.style.width = '100%';
                    mockLoginButton.style.marginTop = 'var(--space-3)';
                    mockLoginButton.textContent = 'Continue as Guest';
                    mockLoginButton.addEventListener('click', () => {
                        setUser({
                            id: 'guest',
                            name: 'Guest User',
                            email: 'guest@example.com',
                            role: 'guest'
                        });
                    });

                    const cardFooter = loginCard.querySelector('.card-footer');
                    if (cardFooter) {
                        cardFooter.insertBefore(mockLoginButton, cardFooter.firstChild);
                    }
                }
            });
        }
    }

    // Render user info in the dashboard
    function renderUserInfo() {
        const dashboardHeader = document.querySelector('.header-content');
        if (!dashboardHeader) {
            // Create header if it doesn't exist
            const dashboardContainer = document.getElementById('dashboard-container');
            if (dashboardContainer) {
                dashboardContainer.innerHTML = `
                    <header class="header">
                        <div class="header-content">
                            <div class="logo">
                                <i class="fas fa-robot logo-icon"></i>
                                <span>Data Insights Chatbot</span>
                            </div>
                            <nav class="nav">
                                <div class="user-menu">
                                    <button id="user-menu-button" class="user-button">
                                        <div class="user-avatar">${currentUser?.name?.charAt(0) || 'U'}</div>
                                        <span class="user-name">${currentUser?.name || 'User'}</span>
                                        <i class="fas fa-chevron-down"></i>
                                    </button>
                                    <div id="user-dropdown" class="dropdown-menu">
                                        <a href="#" class="dropdown-item">
                                            <i class="fas fa-user"></i>
                                            <span>Profile</span>
                                        </a>
                                        <a href="#" class="dropdown-item">
                                            <i class="fas fa-cog"></i>
                                            <span>Settings</span>
                                        </a>
                                        <a href="#" id="logout-button" class="dropdown-item">
                                            <i class="fas fa-sign-out-alt"></i>
                                            <span>Logout</span>
                                        </a>
                                    </div>
                                </div>
                            </nav>
                        </div>
                    </header>
                    <main class="main">
                        <div class="container">
                            <div id="dashboard-content"></div>
                        </div>
                    </main>
                    <footer class="footer">
                        <div class="container" style="text-align: center;">
                            <p>Data Insights Chatbot &copy; 2025. All rights reserved.</p>
                        </div>
                    </footer>
                `;

                // Initialize dashboard content
                initializeDashboard();

                // Add event listeners for user menu
                setupUserMenu();
            }
        } else {
            // Update existing user info
            const userAvatar = dashboardHeader.querySelector('.user-avatar');
            const userName = dashboardHeader.querySelector('.user-name');

            if (userAvatar) {
                userAvatar.textContent = currentUser?.name?.charAt(0) || 'U';
            }

            if (userName) {
                userName.textContent = currentUser?.name || 'User';
            }

            // Setup user menu if not already setup
            setupUserMenu();
        }
    }

    // Setup user menu dropdown
    function setupUserMenu() {
        const userMenuButton = document.getElementById('user-menu-button');
        const userDropdown = document.getElementById('user-dropdown');
        const logoutButton = document.getElementById('logout-button');

        if (userMenuButton && userDropdown) {
            userMenuButton.addEventListener('click', () => {
                userDropdown.classList.toggle('active');
            });

            // Close dropdown when clicking outside
            document.addEventListener('click', (event) => {
                if (!userMenuButton.contains(event.target) && !userDropdown.contains(event.target)) {
                    userDropdown.classList.remove('active');
                }
            });
        }

        if (logoutButton) {
            logoutButton.addEventListener('click', async (e) => {
                e.preventDefault();
                await logout();
            });
        }
    }

    // Initialize dashboard content
    function initializeDashboard() {
        const dashboardContent = document.getElementById('dashboard-content');
        if (!dashboardContent) return;

        // This is a placeholder for the dashboard content
        // The actual implementation will be in the app.js file
    }

    // Check if user is logged in
    function isLoggedIn() {
        return !!currentUser;
    }

    // Get current user
    function getUser() {
        return currentUser;
    }

    // Set user and notify listeners
    function setUser(user) {
        currentUser = user;

        if (user) {
            localStorage.setItem('user', JSON.stringify(user));
        } else {
            localStorage.removeItem('user');
        }

        notifyListeners();
        renderAuthUI();
    }

    // Logout user
    async function logout() {
        try {
            await ApiService.logout();
        } catch (error) {
            logger.error('Logout error', error);
        } finally {
            setUser(null);
        }
    }

    // Add auth state change listener
    function addAuthListener(listener) {
        if (typeof listener === 'function' && !authListeners.includes(listener)) {
            authListeners.push(listener);
        }
    }

    // Remove auth state change listener
    function removeAuthListener(listener) {
        const index = authListeners.indexOf(listener);
        if (index !== -1) {
            authListeners.splice(index, 1);
        }
    }

    // Notify all listeners of auth state change
    function notifyListeners() {
        for (const listener of authListeners) {
            try {
                listener(currentUser);
            } catch (error) {
                logger.error('Auth listener error', error);
            }
        }
    }

    return {
        init,
        isLoggedIn,
        getUser,
        setUser,
        logout,
        addAuthListener,
        removeAuthListener
    };
})();