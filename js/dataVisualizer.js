// Data Visualizer Service
const DataVisualizerService = (function() {
    // Initialize Data Visualizer
    function init() {
        return Promise.resolve();
    }

    // Render a table from data
    function renderTable(container, data, options = {}) {
        if (!container || !data || !Array.isArray(data) || data.length === 0) {
            container.innerHTML = '<div class="alert alert-info">No data available</div>';
            return;
        }

        const tableContainer = document.createElement('div');
        tableContainer.className = 'table-container';

        const table = document.createElement('table');
        table.className = 'table';

        if (options.striped) {
            table.classList.add('table-striped');
        }

        // Create table header
        const thead = document.createElement('thead');
        const headerRow = document.createElement('tr');

        const columns = options.columns || Object.keys(data[0]);

        columns.forEach(column => {
            const th = document.createElement('th');
            th.textContent = options.headerFormatter ? options.headerFormatter(column) : column;
            headerRow.appendChild(th);
        });

        thead.appendChild(headerRow);
        table.appendChild(thead);

        // Create table body
        const tbody = document.createElement('tbody');

        data.forEach(row => {
            const tr = document.createElement('tr');

            columns.forEach(column => {
                const td = document.createElement('td');
                const value = row[column];

                if (options.cellFormatter) {
                    td.innerHTML = options.cellFormatter(value, column, row);
                } else {
                    td.textContent = value !== null && value !== undefined ? value : '';
                }

                tr.appendChild(td);
            });

            tbody.appendChild(tr);
        });

        table.appendChild(tbody);
        tableContainer.appendChild(table);

        // Add pagination if needed
        if (options.pagination && options.pagination.enabled) {
            const paginationContainer = renderPagination(
                options.pagination.currentPage || 1,
                options.pagination.totalPages || 1,
                options.pagination.onPageChange
            );

            tableContainer.appendChild(paginationContainer);
        }

        container.innerHTML = '';
        container.appendChild(tableContainer);
    }

    // Render pagination controls
    function renderPagination(currentPage, totalPages, onPageChange) {
        const paginationContainer = document.createElement('div');
        paginationContainer.className = 'pagination';

        // Previous button
        const prevButton = document.createElement('button');
        prevButton.className = 'pagination-button';
        prevButton.innerHTML = '<i class="fas fa-chevron-left"></i>';
        prevButton.disabled = currentPage <= 1;
        prevButton.addEventListener('click', () => {
            if (currentPage > 1 && typeof onPageChange === 'function') {
                onPageChange(currentPage - 1);
            }
        });

        // Page buttons
        const pageButtonsContainer = document.createElement('div');
        pageButtonsContainer.className = 'pagination-buttons';

        let startPage = Math.max(1, currentPage - 2);
        let endPage = Math.min(totalPages, startPage + 4);

        if (endPage - startPage < 4) {
            startPage = Math.max(1, endPage - 4);
        }

        for (let i = startPage; i <= endPage; i++) {
            const pageButton = document.createElement('button');
            pageButton.className = 'pagination-button';
            if (i === currentPage) {
                pageButton.classList.add('active');
            }
            pageButton.textContent = i;
            pageButton.addEventListener('click', () => {
                if (i !== currentPage && typeof onPageChange === 'function') {
                    onPageChange(i);
                }
            });

            pageButtonsContainer.appendChild(pageButton);
        }

        // Next button
        const nextButton = document.createElement('button');
        nextButton.className = 'pagination-button';
        nextButton.innerHTML = '<i class="fas fa-chevron-right"></i>';
        nextButton.disabled = currentPage >= totalPages;
        nextButton.addEventListener('click', () => {
            if (currentPage < totalPages && typeof onPageChange === 'function') {
                onPageChange(currentPage + 1);
            }
        });

        paginationContainer.appendChild(prevButton);
        paginationContainer.appendChild(pageButtonsContainer);
        paginationContainer.appendChild(nextButton);

        return paginationContainer;
    }

    // Render a chart using Plotly
    function renderChart(container, chartConfig) {
        if (!container || !chartConfig) {
            container.innerHTML = '<div class="alert alert-info">No chart configuration provided</div>';
            return;
        }

        // Make sure container has a minimum height
        container.style.minHeight = '400px';

        // Create chart
        Plotly.newPlot(container, chartConfig.data, chartConfig.layout, chartConfig.config);
    }

    // Create a bar chart
    function createBarChart(data, options = {}) {
        const defaultOptions = {
            x: Object.keys(data[0])[0],
            y: Object.keys(data[0])[1],
            title: 'Bar Chart',
            xAxisTitle: options.x || Object.keys(data[0])[0],
            yAxisTitle: options.y || Object.keys(data[0])[1],
            colorScale: 'Blues',
            orientation: 'v'
        };

        const chartOptions = { ...defaultOptions, ...options };

        const chartData = [{
            x: chartOptions.orientation === 'v' ? data.map(d => d[chartOptions.x]) : data.map(d => d[chartOptions.y]),
            y: chartOptions.orientation === 'v' ? data.map(d => d[chartOptions.y]) : data.map(d => d[chartOptions.x]),
            type: 'bar',
            marker: {
                color: data.map((d, i) => i),
                colorscale: chartOptions.colorScale
            }
        }];

        const layout = {
            title: chartOptions.title,
            xaxis: {
                title: chartOptions.orientation === 'v' ? chartOptions.xAxisTitle : chartOptions.yAxisTitle
            },
            yaxis: {
                title: chartOptions.orientation === 'v' ? chartOptions.yAxisTitle : chartOptions.xAxisTitle
            },
            margin: {
                l: 50,
                r: 50,
                b: 50,
                t: 50,
                pad: 4
            }
        };

        const config = {
            responsive: true,
            displayModeBar: true,
            displaylogo: false,
            modeBarButtonsToRemove: ['lasso2d', 'select2d']
        };

        return { data: chartData, layout, config };
    }

    // Create a line chart
    function createLineChart(data, options = {}) {
        const defaultOptions = {
            x: Object.keys(data[0])[0],
            y: Object.keys(data[0])[1],
            title: 'Line Chart',
            xAxisTitle: options.x || Object.keys(data[0])[0],
            yAxisTitle: options.y || Object.keys(data[0])[1],
            colorScale: 'Blues',
            mode: 'lines+markers'
        };

        const chartOptions = { ...defaultOptions, ...options };

        const chartData = [{
            x: data.map(d => d[chartOptions.x]),
            y: data.map(d => d[chartOptions.y]),
            type: 'scatter',
            mode: chartOptions.mode,
            line: {
                color: chartOptions.lineColor || '#3B82F6',
                width: 3
            },
            marker: {
                size: 8,
                color: chartOptions.markerColor || '#3B82F6',
                line: {
                    width: 2,
                    color: 'white'
                }
            }
        }];

        const layout = {
            title: chartOptions.title,
            xaxis: {
                title: chartOptions.xAxisTitle
            },
            yaxis: {
                title: chartOptions.yAxisTitle
            },
            margin: {
                l: 50,
                r: 50,
                b: 50,
                t: 50,
                pad: 4
            }
        };

        const config = {
            responsive: true,
            displayModeBar: true,
            displaylogo: false,
            modeBarButtonsToRemove: ['lasso2d', 'select2d']
        };

        return { data: chartData, layout, config };
    }

    // Create a pie chart
    function createPieChart(data, options = {}) {
        const defaultOptions = {
            labels: Object.keys(data[0])[0],
            values: Object.keys(data[0])[1],
            title: 'Pie Chart',
            colorScale: 'Blues'
        };

        const chartOptions = { ...defaultOptions, ...options };

        const chartData = [{
            labels: data.map(d => d[chartOptions.labels]),
            values: data.map(d => d[chartOptions.values]),
            type: 'pie',
            marker: {
                colors: data.map((d, i) => i),
                colorscale: chartOptions.colorScale
            },
            textinfo: 'label+percent',
            insidetextorientation: 'radial'
        }];

        const layout = {
            title: chartOptions.title,
            margin: {
                l: 50,
                r: 50,
                b: 50,
                t: 50,
                pad: 4
            }
        };

        const config = {
            responsive: true,
            displayModeBar: true,
            displaylogo: false,
            modeBarButtonsToRemove: ['lasso2d', 'select2d']
        };

        return { data: chartData, layout, config };
    }

    // Format data for charting
    function formatDataForChart(data, chartType, options = {}) {
        switch (chartType.toLowerCase()) {
            case 'bar':
                return createBarChart(data, options);
            case 'line':
                return createLineChart(data, options);
            case 'pie':
                return createPieChart(data, options);
            default:
                return createBarChart(data, options);
        }
    }

    return {
        init,
        renderTable,
        renderChart,
        createBarChart,
        createLineChart,
        createPieChart,
        formatDataForChart
    };
})();