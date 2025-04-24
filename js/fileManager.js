// File Manager Service
const FileManagerService = (function() {
    let uploadedFiles = [];
    let fileListeners = [];

    // Initialize File Manager
    function init() {
        return refreshFileList();
    }

    // Refresh the list of uploaded files
    async function refreshFileList() {
        // Currently, there's no direct endpoint to get the list of uploaded files
        // For demonstration purposes, we'll maintain a local list
        return uploadedFiles;
    }

    // Upload files
    async function uploadFiles(files) {
        if (!files || files.length === 0) {
            return { success: false, message: 'No files selected' };
        }

        // Validate files
        for (const file of files) {
            // Check file size
            if (file.size > CONFIG.MAX_FILE_SIZE) {
                return {
                    success: false,
                    message: `File "${file.name}" exceeds the maximum size limit of ${formatFileSize(CONFIG.MAX_FILE_SIZE)}`
                };
            }

            // Check file type
            const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
            if (!CONFIG.ALLOWED_FILE_TYPES.includes(fileExtension)) {
                return {
                    success: false,
                    message: `File "${file.name}" has an unsupported file type. Allowed types: ${CONFIG.ALLOWED_FILE_TYPES.join(', ')}`
                };
            }
        }

        try {
            const formData = new FormData();
            for (let i = 0; i < files.length; i++) {
                formData.append(`file${i}`, files[i]);
            }

            const response = await ApiService.uploadFile(formData);

            if (response && response.type === 'text') {
                // Add files to the local list
                for (const file of files) {
                    uploadedFiles.push({
                        name: file.name,
                        size: file.size,
                        type: file.type,
                        uploadedAt: new Date()
                    });
                }

                // Notify listeners
                notifyListeners();

                return { success: true, message: Array.isArray(response.text) ? response.text.join('\n') : response.text };
            } else {
                throw new Error(response?.error || 'Failed to upload files');
            }
        } catch (error) {
            logger.error('Upload files error', error);
            return { success: false, message: error.message || 'Failed to upload files' };
        }
    }

    // Delete a file
    async function deleteFile(fileName) {
        try {
            const response = await ApiService.deleteFile(fileName);

            if (response && response.type === 'text') {
                // Remove file from the local list
                uploadedFiles = uploadedFiles.filter(file => file.name !== fileName);

                // Notify listeners
                notifyListeners();

                return { success: true, message: response.text };
            } else {
                throw new Error(response?.error || 'Failed to delete file');
            }
        } catch (error) {
            logger.error('Delete file error', error);
            return { success: false, message: error.message || 'Failed to delete file' };
        }
    }

    // Delete all files
    async function deleteAllFiles() {
        try {
            const response = await ApiService.deleteAllFiles();

            if (response && response.type === 'text') {
                // Clear the local list
                uploadedFiles = [];

                // Notify listeners
                notifyListeners();

                return { success: true, message: response.text };
            } else {
                throw new Error(response?.error || 'Failed to delete all files');
            }
        } catch (error) {
            logger.error('Delete all files error', error);
            return { success: false, message: error.message || 'Failed to delete all files' };
        }
    }

    // Render file manager UI
    function render(container) {
        if (!container) return;

        container.innerHTML = `
            <div class="card">
                <div class="card-header">
                    <h3 class="card-title">Data Files</h3>
                    <p class="card-subtitle">Upload CSV or Excel files to analyze</p>
                </div>
                <div class="card-body">
                    <div class="file-upload">
                        <label for="file-upload-input" class="file-upload-label">
                            <i class="fas fa-cloud-upload-alt file-upload-icon"></i>
                            <div class="file-upload-text">
                                <strong>Choose files</strong> or drag them here
                            </div>
                            <div class="file-upload-text">
                                CSV, XLS, XLSX up to 10MB
                            </div>
                        </label>
                        <input type="file" id="file-upload-input" multiple accept=".csv,.xls,.xlsx">
                    </div>
                    
                    <div id="file-list" class="file-list">
                        ${renderFileList()}
                    </div>
                </div>
                <div class="card-footer">
                    <button id="delete-all-files" class="btn btn-danger" ${uploadedFiles.length === 0 ? 'disabled' : ''}>
                        <i class="fas fa-trash btn-icon"></i>
                        Delete All Files
                    </button>
                </div>
            </div>
        `;

        // Add event listeners
        setupFileUpload(container);
    }

    // Render file list
    function renderFileList() {
        if (uploadedFiles.length === 0) {
            return '<div class="file-list-empty">No files uploaded yet</div>';
        }

        return uploadedFiles.map(file => `
            <div class="file-item" data-name="${file.name}">
                <i class="fas ${getFileIcon(file.name)} file-item-icon"></i>
                <div class="file-item-name">${file.name}</div>
                <div class="file-item-size">${formatFileSize(file.size)}</div>
                <button class="file-item-remove" data-name="${file.name}" title="Delete file">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `).join('');
    }

    // Get file icon based on file extension
    function getFileIcon(fileName) {
        const extension = fileName.split('.').pop().toLowerCase();

        switch (extension) {
            case 'csv':
                return 'fa-file-csv';
            case 'xls':
            case 'xlsx':
                return 'fa-file-excel';
            default:
                return 'fa-file';
        }
    }

    // Setup file upload
    function setupFileUpload(container) {
        const fileUploadInput = container.querySelector('#file-upload-input');
        const fileUploadLabel = container.querySelector('.file-upload-label');
        const fileList = container.querySelector('#file-list');
        const deleteAllButton = container.querySelector('#delete-all-files');

        if (fileUploadInput && fileUploadLabel) {
            // File input change event
            fileUploadInput.addEventListener('change', async () => {
                if (fileUploadInput.files.length > 0) {
                    setLoading(fileUploadLabel, true);

                    const result = await uploadFiles(fileUploadInput.files);

                    if (result.success) {
                        showToast(result.message, 'success');

                        // Update file list
                        if (fileList) {
                            fileList.innerHTML = renderFileList();
                            setupFileDeleteButtons(container);
                        }

                        // Enable delete all button
                        if (deleteAllButton) {
                            deleteAllButton.disabled = uploadedFiles.length === 0;
                        }
                    } else {
                        showToast(result.message, 'error');
                    }

                    setLoading(fileUploadLabel, false);

                    // Reset the input to allow uploading the same file again
                    fileUploadInput.value = '';
                }
            });

            // Drag and drop
            fileUploadLabel.addEventListener('dragover', (e) => {
                e.preventDefault();
                fileUploadLabel.classList.add('dragover');
            });

            fileUploadLabel.addEventListener('dragleave', () => {
                fileUploadLabel.classList.remove('dragover');
            });

            fileUploadLabel.addEventListener('drop', (e) => {
                e.preventDefault();
                fileUploadLabel.classList.remove('dragover');

                if (e.dataTransfer.files.length > 0) {
                    fileUploadInput.files = e.dataTransfer.files;
                    fileUploadInput.dispatchEvent(new Event('change'));
                }
            });
        }

        // Setup delete buttons
        setupFileDeleteButtons(container);

        // Delete all files
        if (deleteAllButton) {
            deleteAllButton.addEventListener('click', async () => {
                if (uploadedFiles.length === 0) return;

                // Show confirmation dialog
                showModal('Delete All Files', 'Are you sure you want to delete all files? This action cannot be undone.', null, [
                    {
                        text: 'Cancel',
                        type: 'outline',
                        onClick: (closeModal) => closeModal()
                    },
                    {
                        text: 'Delete All',
                        type: 'danger',
                        onClick: async (closeModal) => {
                            setLoading(deleteAllButton, true);

                            const result = await deleteAllFiles();

                            if (result.success) {
                                showToast(result.message, 'success');

                                // Update file list
                                if (fileList) {
                                    fileList.innerHTML = renderFileList();
                                }

                                // Disable delete all button
                                deleteAllButton.disabled = true;
                            } else {
                                showToast(result.message, 'error');
                            }

                            setLoading(deleteAllButton, false);
                            closeModal();
                        }
                    }
                ]);
            });
        }
    }

    // Setup file delete buttons
    function setupFileDeleteButtons(container) {
        const deleteButtons = container.querySelectorAll('.file-item-remove');
        const fileList = container.querySelector('#file-list');
        const deleteAllButton = container.querySelector('#delete-all-files');

        deleteButtons.forEach(button => {
            button.addEventListener('click', async () => {
                const fileName = button.dataset.name;
                if (!fileName) return;

                const fileItem = container.querySelector(`.file-item[data-name="${fileName}"]`);
                if (fileItem) {
                    setLoading(fileItem, true);

                    const result = await deleteFile(fileName);

                    if (result.success) {
                        showToast(result.message, 'success');

                        // Update file list
                        if (fileList) {
                            fileList.innerHTML = renderFileList();
                            setupFileDeleteButtons(container);
                        }

                        // Disable delete all button if no files
                        if (deleteAllButton) {
                            deleteAllButton.disabled = uploadedFiles.length === 0;
                        }
                    } else {
                        showToast(result.message, 'error');
                        setLoading(fileItem, false);
                    }
                }
            });
        });
    }

    // Show modal with buttons
    function showModal(title, content, onClose, buttons) {
        const modalOverlay = document.querySelector('.modal-overlay') ||
            document.body.appendChild(createElement('div', { className: 'modal-overlay' }));

        const modalId = generateUniqueId();

        // Create modal body
        let modalBody;
        if (typeof content === 'string') {
            modalBody = createElement('div', { innerHTML: content });
        } else {
            modalBody = content;
        }

        // Create modal footer with buttons
        let modalFooter;
        if (buttons && buttons.length > 0) {
            modalFooter = createElement('div', { className: 'modal-footer' });

            buttons.forEach(button => {
                const btn = createElement('button', {
                    className: `btn btn-${button.type || 'primary'}`,
                    textContent: button.text
                });

                if (typeof button.onClick === 'function') {
                    btn.addEventListener('click', () => button.onClick(() => closeModal(modalId, onClose)));
                }

                modalFooter.appendChild(btn);
            });
        }

        const modal = createElement('div', { className: 'modal', id: `modal-${modalId}` }, [
            createElement('div', { className: 'modal-header' }, [
                createElement('h3', { className: 'modal-title', textContent: title }),
                createElement('button', {
                    className: 'modal-close',
                    innerHTML: '&times;',
                    onclick: () => closeModal(modalId, onClose)
                })
            ]),
            createElement('div', { className: 'modal-body' }, [modalBody]),
            modalFooter
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

    // Add file change listener
    function addFileListener(listener) {
        if (typeof listener === 'function' && !fileListeners.includes(listener)) {
            fileListeners.push(listener);
        }
    }

    // Remove file change listener
    function removeFileListener(listener) {
        const index = fileListeners.indexOf(listener);
        if (index !== -1) {
            fileListeners.splice(index, 1);
        }
    }

    // Notify all listeners of file changes
    function notifyListeners() {
        for (const listener of fileListeners) {
            try {
                listener(uploadedFiles);
            } catch (error) {
                logger.error('File listener error', error);
            }
        }
    }

    return {
        init,
        refreshFileList,
        uploadFiles,
        deleteFile,
        deleteAllFiles,
        render,
        addFileListener,
        removeFileListener,
        getFiles: () => [...uploadedFiles]
    };
})();