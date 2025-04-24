// Configuration Variables
const CONFIG = {
    API_BASE_URL: '', // Empty string means relative to current path
    DEFAULT_HEADERS: {
        'Content-Type': 'application/json'
    },
    TOAST_TIMEOUT: 3000,
    MAX_FILE_SIZE: 10 * 1024 * 1024, // 10MB
    ALLOWED_FILE_TYPES: ['.csv', '.xlsx', '.xls'],
    ANIMATION_DURATION: 300,
    DEBUG: false // Set to true to enable debug logging
};

// Logging utility
const logger = {
    log: function(message) {
        if (CONFIG.DEBUG) {
            console.log(`[LOG] ${message}`);
        }
    },
    error: function(message, error) {
        if (CONFIG.DEBUG) {
            console.error(`[ERROR] ${message}`, error);
        }
    },
    warn: function(message) {
        if (CONFIG.DEBUG) {
            console.warn(`[WARN] ${message}`);
        }
    },
    info: function(message) {
        if (CONFIG.DEBUG) {
            console.info(`[INFO] ${message}`);
        }
    }
};

// Utility Functions
function formatDate(date) {
    return new Date(date).toLocaleString();
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function debounce(func, wait) {
    let timeout;
    return function(...args) {
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(this, args), wait);
    };
}

function sanitizeHTML(str) {
    const temp = document.createElement('div');
    temp.textContent = str;
    return temp.innerHTML;
}

function generateUniqueId() {
    return 'id-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
}

// DOM Utility Functions
function createElement(tag, attributes = {}, children = []) {
    const element = document.createElement(tag);

    for (const [key, value] of Object.entries(attributes)) {
        if (key === 'className') {
            element.className = value;
        } else if (key === 'textContent') {
            element.textContent = value;
        } else if (key === 'innerHTML') {
            element.innerHTML = value;
        } else if (key === 'style' && typeof value === 'object') {
            Object.assign(element.style, value);
        } else if (key.startsWith('on') && typeof value === 'function') {
            element.addEventListener(key.slice(2).toLowerCase(), value);
        } else {
            element.setAttribute(key, value);
        }
    }

    for (const child of children) {
        if (typeof child === 'string') {
            element.appendChild(document.createTextNode(child));
        } else if (child instanceof Node) {
            element.appendChild(child);
        }
    }

    return element;
}

function showElement(element) {
    if (typeof element === 'string') {
        element = document.querySelector(element);
    }
    if (element) {
        element.classList.remove('hidden');
    }
}

function hideElement(element) {
    if (typeof element === 'string') {
        element = document.querySelector(element);
    }
    if (element) {
        element.classList.add('hidden');
    }
}

function toggleElement(element) {
    if (typeof element === 'string') {
        element = document.querySelector(element);
    }
    if (element) {
        element.classList.toggle('hidden');
    }
}

function setLoading(element, isLoading) {
    if (typeof element === 'string') {
        element = document.querySelector(element);
    }
    if (element) {
        if (isLoading) {
            element.classList.add('loading');
            if (element.tagName === 'BUTTON') {
                element.disabled = true;
            }
        } else {
            element.classList.remove('loading');
            if (element.tagName === 'BUTTON') {
                element.disabled = false;
            }
        }
    }
}

// Show a toast notification
function showToast(message, type = 'info', duration = CONFIG.TOAST_TIMEOUT) {
    const toastContainer = document.querySelector('.toast-container') ||
        document.body.appendChild(createElement('div', { className: 'toast-container' }));

    const iconMap = {
        success: 'fa-check-circle',
        error: 'fa-exclamation-circle',
        warning: 'fa-exclamation-triangle',
        info: 'fa-info-circle'
    };

    const toastId = generateUniqueId();

    const toast = createElement('div', {
        className: `toast toast-${type} fade-in-toast`,
        id: toastId
    }, [
        createElement('i', { className: `fas ${iconMap[type]} toast-icon` }),
        createElement('div', { className: 'toast-content' }, [
            createElement('div', { className: 'toast-message', textContent: message })
        ]),
        createElement('button', {
            className: 'toast-close',
            title: 'Close',
            innerHTML: '&times;',
            onclick: () => closeToast(toastId)
        })
    ]);

    toastContainer.appendChild(toast);

    if (duration > 0) {
        setTimeout(() => closeToast(toastId), duration);
    }

    return toastId;
}

function closeToast(toastId) {
    const toast = document.getElementById(toastId);
    if (toast) {
        toast.classList.add('fade-out-toast');
        toast.classList.remove('fade-in-toast');
        setTimeout(() => {
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        }, 300);
    }
}

// Show modal
function showModal(title, content, onClose) {
    const modalOverlay = document.querySelector('.modal-overlay') ||
        document.body.appendChild(createElement('div', { className: 'modal-overlay' }));

    const modalId = generateUniqueId();

    const modal = createElement('div', { className: 'modal', id: `modal-${modalId}` }, [
        createElement('div', { className: 'modal-header' }, [
            createElement('h3', { className: 'modal-title', textContent: title }),
            createElement('button', {
                className: 'modal-close',
                innerHTML: '&times;',
                onclick: () => closeModal(modalId, onClose)
            })
        ]),
        createElement('div', { className: 'modal-body' }, [
            typeof content === 'string' ? createElement('div', { innerHTML: content }) : content
        ])
    ]);

    modalOverlay.innerHTML = '';
    modalOverlay.appendChild(modal);

    setTimeout(() => {
        modalOverlay.classList.add('active');
    }, 10);

    modalOverlay.addEventListener('click', (e) => {
        if (e.target === modalOverlay) {
            closeModal(modalId, onClose);
        }
    });

    return modalId;
}

function closeModal(modalId, onClose) {
    const modalOverlay = document.querySelector('.modal-overlay');
    if (modalOverlay) {
        modalOverlay.classList.remove('active');
        setTimeout(() => {
            modalOverlay.innerHTML = '';
            if (typeof onClose === 'function') {
                onClose();
            }
        }, CONFIG.ANIMATION_DURATION);
    }
}