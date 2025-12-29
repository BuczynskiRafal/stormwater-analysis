document.addEventListener('DOMContentLoaded', (event) => {
    const tables = document.querySelectorAll('.data-table');
    tables.forEach((table) => {
        initializeDataTable(table);
    });
});

document.addEventListener('DOMContentLoaded', function() {
    let tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-toggle="tooltip"]'))
    let tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl, {
            animation: true,
            delay: { "show": 800, "hide": 100 },
            placement: 'auto',
            html: true
        })
    });

    setTimeout(function() {
        tooltipList.forEach(function(tooltip) {
            tooltip.hide();
        });
    }, 5000);
});

const initializeDataTable = function(table) {
    $(table)?.DataTable?.({
        ordering: true,
        order: [[1, 'desc']],
        searching: false,
        paging: false,
        info: false,
        fixedHeader: false, // Disable DataTables fixed header
        initComplete: function () {
            const label = $(table).parents('.dataTables_wrapper').find('label');
            let labelHtml = label.html();
            if (labelHtml) {
                label.html(labelHtml.replace('Search:', 'Search: '));
            }

            // Reset any DataTables top styles that might interfere
            $(table).find('thead th').each(function() {
                this.style.removeProperty('top');
            });
        }
    });
}

document.addEventListener('DOMContentLoaded', function() {
    const accordionHeaders = document.querySelectorAll('.accordion-header');
    const accordionItems = document.querySelectorAll('.accordion-item');

    // Initialize drag to scroll for all table containers
    function initializeDragToScroll() {
        const containers = document.querySelectorAll('.scrollable-table-container');

        containers.forEach(container => {
            let isDown = false;
            let startX;
            let scrollLeft;
            let hasScrolled = false;

            // Add cursor styles
            container.style.cursor = 'grab';

            container.addEventListener('mousedown', (e) => {
                isDown = true;
                container.style.cursor = 'grabbing';
                startX = e.pageX - container.offsetLeft;
                scrollLeft = container.scrollLeft;
                container.style.userSelect = 'none'; // Prevent text selection
            });

            container.addEventListener('mouseleave', () => {
                isDown = false;
                container.style.cursor = 'grab';
            });

            container.addEventListener('mouseup', () => {
                isDown = false;
                container.style.cursor = 'grab';
            });

            container.addEventListener('mousemove', (e) => {
                if (!isDown) return;
                e.preventDefault();
                const x = e.pageX - container.offsetLeft;
                const walk = (x - startX) * 2; // *2 for faster scrolling
                container.scrollLeft = scrollLeft - walk;

                // Hide badge after first horizontal drag scroll (2x faster - 200ms)
                if (!hasScrolled && Math.abs(walk) > 5) {
                    hasScrolled = true;
                    container.classList.add('scrolled');
                    setTimeout(() => {
                        container.classList.add('hide-badge');
                    }, 200);
                }
            });

            // Also hide badge on regular scroll - detect if horizontal or vertical
            container.addEventListener('scroll', () => {
                if (!hasScrolled) {
                    hasScrolled = true;
                    container.classList.add('scrolled');
                    // For horizontal scroll, fade out in 200ms, for vertical - immediately
                    container.classList.add('hide-badge');
                }
            });
        });
    }

    // Initialize on page load
    initializeDragToScroll();

    // Hide badges immediately on vertical page scroll
    window.addEventListener('scroll', () => {
        document.querySelectorAll('.scrollable-table-container:not(.hide-badge)').forEach(container => {
            container.classList.add('hide-badge');
        });
    });

    function handleStickyHeaders() {
        const header = document.querySelector('header');
        const headerHeight = header ? header.offsetHeight : 0;
        const scrollTop = window.pageYOffset || document.documentElement.scrollTop;

        // Get current position of navigation header
        const headerRect = header ? header.getBoundingClientRect() : null;
        const navBottomPosition = headerRect ? Math.max(0, headerRect.bottom) : 0;

        accordionItems.forEach((item) => {
            const accordionHeader = item.querySelector('.accordion-header');
            const collapse = item.querySelector('.accordion-collapse');
            const itemRect = item.getBoundingClientRect();
            const itemTop = itemRect.top + scrollTop;
            const itemBottom = itemRect.bottom + scrollTop;

            // Check if accordion is expanded
            const isExpanded = collapse.classList.contains('show');

            if (isExpanded) {
                // Check if we should make header sticky
                const shouldBeSticky = scrollTop > (itemTop - headerHeight) &&
                                     scrollTop < (itemBottom - headerHeight - 100);

                if (shouldBeSticky) {
                    accordionHeader.classList.add('sticky');
                    item.classList.add('header-sticky');

                    // Position sticky header to follow navigation bar smoothly
                    accordionHeader.style.top = navBottomPosition + 'px';

                    // Add appropriate class based on nav position
                    if (navBottomPosition > 0) {
                        accordionHeader.classList.add('sticky-with-nav');
                        accordionHeader.classList.remove('sticky-top');
                    } else {
                        accordionHeader.classList.add('sticky-top');
                        accordionHeader.classList.remove('sticky-with-nav');
                    }

                    // Keep original position and width
                    const itemCurrentRect = item.getBoundingClientRect();
                    accordionHeader.style.left = itemCurrentRect.left + 'px';
                    accordionHeader.style.width = item.offsetWidth + 'px';

                    // Make table headers stick to accordion header (same approach as accordion to nav)
                    const tableHeaders = item.querySelectorAll('.scrollable-table-container .table thead th');
                    const accordionHeaderHeight = accordionHeader.offsetHeight || 60;

                    tableHeaders.forEach((th) => {
                        const tableRect = th.closest('table').getBoundingClientRect();
                        const tableTop = tableRect.top + scrollTop;
                        const tableBottom = tableRect.bottom + scrollTop;

                        // Check if table header should be sticky (when table reaches accordion)
                        const tableShouldBeSticky = scrollTop > (tableTop - headerHeight - accordionHeaderHeight) &&
                                                  scrollTop < (tableBottom - headerHeight - accordionHeaderHeight - 50);

                        if (tableShouldBeSticky) {
                            // Create or update fixed header clone
                            let fixedTable = item.querySelector('.fixed-table-header');
                            if (!fixedTable) {
                                const table = th.closest('table');
                                fixedTable = table.cloneNode(true);

                                // Preserve original classes and add our fixed class
                                fixedTable.className = table.className + ' fixed-table-header';

                                // Keep only thead, remove tbody
                                const tbody = fixedTable.querySelector('tbody');
                                if (tbody) tbody.remove();

                                // Get container dimensions for clipping
                                const container = th.closest('.scrollable-table-container');
                                const containerRect = container.getBoundingClientRect();
                                const accordionItem = th.closest('.accordion-item');
                                const accordionItemRect = accordionItem.getBoundingClientRect();
                                const tableRect = table.getBoundingClientRect();
                                const scrollLeft = container.scrollLeft;

                                // Use container width directly (no clipping)
                                const maxWidth = containerRect.width;


                                // Position at original container location but clip to accordion bounds
                                fixedTable.style.cssText = `
                                    position: fixed !important;
                                    top: ${navBottomPosition + accordionHeaderHeight}px;
                                    left: ${containerRect.left}px;
                                    width: ${Math.min(containerRect.width, maxWidth)}px;
                                    max-width: ${maxWidth}px;
                                    z-index: 1018;
                                    margin: 0 !important;
                                    border-collapse: separate !important;
                                    border-spacing: 0 !important;
                                    overflow: hidden !important;
                                    clip-path: inset(0 0 0 0);
                                `;

                                // Copy computed styles from original table headers to cloned headers
                                const originalHeaders = table.querySelectorAll('thead th');
                                const clonedHeaders = fixedTable.querySelectorAll('thead th');

                                // Force table to use same layout mode as original
                                fixedTable.style.tableLayout = table.style.tableLayout || 'auto';
                                fixedTable.style.width = table.offsetWidth + 'px';

                                // Calculate cumulative offset to match original positioning exactly
                                let cumulativeOffset = 0;

                                originalHeaders.forEach((originalTh, index) => {
                                    if (clonedHeaders[index]) {
                                        const clonedTh = clonedHeaders[index];
                                        const computedStyle = window.getComputedStyle(originalTh);

                                        // Get exact positioning from original table
                                        const originalRect = originalTh.getBoundingClientRect();
                                        const actualWidth = originalRect.width;

                                        // Copy all border/padding/margin properties exactly
                                        clonedTh.style.boxSizing = computedStyle.boxSizing;
                                        clonedTh.style.border = computedStyle.border;
                                        clonedTh.style.borderLeft = computedStyle.borderLeft;
                                        clonedTh.style.borderRight = computedStyle.borderRight;
                                        clonedTh.style.padding = computedStyle.padding;
                                        clonedTh.style.margin = computedStyle.margin;

                                        // Force exact same width as rendered
                                        clonedTh.style.width = actualWidth + 'px';
                                        clonedTh.style.minWidth = actualWidth + 'px';
                                        clonedTh.style.maxWidth = actualWidth + 'px';

                                        // Copy visual styles
                                        clonedTh.style.backgroundColor = computedStyle.backgroundColor;
                                        clonedTh.style.color = computedStyle.color;
                                        clonedTh.style.fontSize = computedStyle.fontSize;
                                        clonedTh.style.fontWeight = computedStyle.fontWeight;
                                        clonedTh.style.textAlign = computedStyle.textAlign;
                                        clonedTh.style.boxShadow = computedStyle.boxShadow;

                                        // Copy DataTables sorting classes
                                        clonedTh.className = originalTh.className;

                                        // Handle scrolling: all columns scroll together
                                        clonedTh.style.transform = `translateX(-${Math.round(scrollLeft)}px)`;

                                        // Enable sorting by forwarding clicks to original header
                                        clonedTh.style.cursor = 'pointer';
                                        clonedTh.addEventListener('click', function(e) {
                                            e.preventDefault();
                                            e.stopPropagation();
                                            // Trigger click on original header for DataTables sorting
                                            originalTh.click();

                                            // Update cloned header classes after sort
                                            setTimeout(() => {
                                                clonedTh.className = originalTh.className;
                                            }, 50);
                                        });

                                        cumulativeOffset += actualWidth;
                                    }
                                });

                                // Create wrapper container for better clipping control
                                let wrapperDiv = item.querySelector('.fixed-table-wrapper');
                                if (!wrapperDiv) {
                                    wrapperDiv = document.createElement('div');
                                    wrapperDiv.className = 'fixed-table-wrapper';
                                    wrapperDiv.style.cssText = `
                                        position: fixed !important;
                                        top: ${navBottomPosition + accordionHeaderHeight}px;
                                        left: ${containerRect.left}px;
                                        width: ${Math.min(containerRect.width, maxWidth)}px;
                                        height: 60px;
                                        z-index: 1018;
                                        overflow: hidden !important;
                                    `;
                                    item.appendChild(wrapperDiv);
                                }

                                // Reset fixed table positioning to be relative to wrapper
                                fixedTable.style.cssText = `
                                    position: relative !important;
                                    top: 0;
                                    left: 0;
                                    width: ${table.offsetWidth}px;
                                    margin: 0 !important;
                                    border-collapse: separate !important;
                                    border-spacing: 0 !important;
                                `;

                                wrapperDiv.appendChild(fixedTable);

                                // Scroll handler for drag-induced scrolling
                                const scrollHandler = () => {
                                    const scrollLeft = container.scrollLeft;
                                    const clonedHeaders = fixedTable.querySelectorAll('thead th');
                                    clonedHeaders.forEach((clonedTh) => {
                                        clonedTh.style.transform = `translateX(-${scrollLeft}px)`;
                                    });
                                };

                                // Wheel handler that mimics drag-to-scroll behavior exactly
                                const wheelHandler = (e) => {
                                    if (Math.abs(e.deltaX) > Math.abs(e.deltaY)) {
                                        e.preventDefault();

                                        // Reduce wheel sensitivity for smoother feel (like drag multiplier)
                                        const walk = e.deltaX * 0.1; // Much smaller steps
                                        const maxScrollLeft = container.scrollWidth - container.clientWidth;
                                        const newScrollLeft = Math.max(0, Math.min(maxScrollLeft, container.scrollLeft + walk));

                                        container.scrollLeft = newScrollLeft;

                                        // Update headers immediately, same as drag does
                                        const clonedHeaders = fixedTable.querySelectorAll('thead th');
                                        clonedHeaders.forEach((clonedTh) => {
                                            clonedTh.style.transform = `translateX(-${newScrollLeft}px)`;
                                        });
                                    }
                                };

                                // Store references for cleanup
                                container._stickyScrollHandler = scrollHandler;
                                container._wheelHandler = wheelHandler;

                                // Add both listeners - scroll for drag, wheel for mouse wheel
                                container.addEventListener('scroll', scrollHandler, { passive: true });
                                container.addEventListener('wheel', wheelHandler, { passive: false });
                            } else {
                                // Update wrapper position and bounds to stay within accordion
                                const wrapperDiv = item.querySelector('.fixed-table-wrapper');
                                if (wrapperDiv) {
                                    const container = th.closest('.scrollable-table-container');
                                    const containerRect = container.getBoundingClientRect();
                                    const accordionItem = th.closest('.accordion-item');
                                    const accordionItemRect = accordionItem.getBoundingClientRect();

                                    wrapperDiv.style.top = (navBottomPosition + accordionHeaderHeight) + 'px';
                                    wrapperDiv.style.left = containerRect.left + 'px';
                                    wrapperDiv.style.width = containerRect.width + 'px';
                                }
                            }

                            // Hide original header when fixed is active
                            th.style.visibility = 'hidden';
                        } else {
                            // Remove fixed header clone and show original
                            const wrapperDiv = item.querySelector('.fixed-table-wrapper');
                            if (wrapperDiv) {
                                // Remove unified scroll listener from this container
                                const container = th.closest('.scrollable-table-container');
                                if (container._stickyScrollHandler) {
                                    container.removeEventListener('scroll', container._stickyScrollHandler);
                                    container._stickyScrollHandler = null;
                                }
                                if (container._wheelHandler) {
                                    container.removeEventListener('wheel', container._wheelHandler);
                                    container._wheelHandler = null;
                                }
                                wrapperDiv.remove();
                            }
                            th.style.visibility = '';
                        }
                    });

                    // Adjust accordion content padding
                    collapse.style.paddingTop = accordionHeaderHeight + 'px';
                } else {
                    accordionHeader.classList.remove('sticky', 'sticky-with-nav', 'sticky-top');
                    item.classList.remove('header-sticky');
                    accordionHeader.style.top = '';
                    accordionHeader.style.left = '';
                    accordionHeader.style.width = '';

                    // Reset table headers - remove clones and show originals
                    const wrapperDivs = item.querySelectorAll('.fixed-table-wrapper');
                    wrapperDivs.forEach(wrapper => {
                        wrapper.remove();
                    });

                    // Clean up unified scroll listeners
                    const containers = item.querySelectorAll('.scrollable-table-container');
                    containers.forEach(container => {
                        if (container._stickyScrollHandler) {
                            container.removeEventListener('scroll', container._stickyScrollHandler);
                            container._stickyScrollHandler = null;
                        }
                        if (container._wheelHandler) {
                            container.removeEventListener('wheel', container._wheelHandler);
                            container._wheelHandler = null;
                        }
                    });

                    const tableHeaders = item.querySelectorAll('.scrollable-table-container .table thead th');
                    tableHeaders.forEach(th => {
                        th.style.visibility = '';
                    });
                    collapse.style.paddingTop = '';
                }
            } else {
                accordionHeader.classList.remove('sticky', 'sticky-with-nav', 'sticky-top');
                item.classList.remove('header-sticky');
                accordionHeader.style.top = '';
                accordionHeader.style.left = '';
                accordionHeader.style.width = '';

                // Reset table headers - remove clones and show originals
                const wrapperDivs = item.querySelectorAll('.fixed-table-wrapper');
                wrapperDivs.forEach(wrapper => {
                    wrapper.remove();
                });

                // Clean up unified scroll listeners
                const containers = item.querySelectorAll('.scrollable-table-container');
                containers.forEach(container => {
                    if (container._stickyScrollHandler) {
                        container.removeEventListener('scroll', container._stickyScrollHandler);
                        container._stickyScrollHandler = null;
                    }
                    if (container._wheelHandler) {
                        container.removeEventListener('wheel', container._wheelHandler);
                        container._wheelHandler = null;
                    }
                });

                const tableHeaders = item.querySelectorAll('.scrollable-table-container .table thead th');
                tableHeaders.forEach(th => {
                    th.style.visibility = '';
                });
                collapse.style.paddingTop = '';
            }
        });
    }

    // Throttle scroll events for better performance
    let ticking = false;
    function requestTick() {
        if (!ticking) {
            requestAnimationFrame(handleStickyHeaders);
            ticking = true;
            setTimeout(() => { ticking = false; }, 16);
        }
    }

    window.addEventListener('scroll', requestTick);
    window.addEventListener('resize', handleStickyHeaders);

    // Handle accordion toggle events
    accordionHeaders.forEach(header => {
        header.addEventListener('click', function() {
            // Small delay to allow accordion animation to start
            setTimeout(() => {
                handleStickyHeaders();
                initializeDragToScroll(); // Re-initialize drag to scroll for new containers
            }, 50);
        });
    });

    // =========================================================================
    // Keyboard Navigation for Detail Pages
    // =========================================================================

    /**
     * Initialize keyboard navigation for detail pages
     * Allows navigating between items using arrow keys
     */
    function initKeyboardNavigation() {
        const leftArrow = document.getElementById('navArrowLeft');
        const rightArrow = document.getElementById('navArrowRight');

        // Only initialize if navigation arrows exist
        if (!leftArrow && !rightArrow) return;

        document.addEventListener('keydown', (event) => {
            // Ignore if user is typing in an input field
            const tagName = event.target.tagName.toLowerCase();
            if (tagName === 'input' || tagName === 'textarea' || event.target.contentEditable === 'true') {
                return;
            }

            if (event.key === 'ArrowLeft') {
                event.preventDefault();
                const link = document.querySelector('#navArrowLeft .nav-link');
                if (link) window.location.href = link.href;
            } else if (event.key === 'ArrowRight') {
                event.preventDefault();
                const link = document.querySelector('#navArrowRight .nav-link');
                if (link) window.location.href = link.href;
            }
        });

        // Enhanced hover effects for navigation areas
        [leftArrow, rightArrow].forEach(arrow => {
            if (arrow) {
                arrow.addEventListener('mouseenter', function() {
                    this.style.opacity = '1';
                });
                arrow.addEventListener('mouseleave', function() {
                    this.style.opacity = '0';
                });
            }
        });
    }

    initKeyboardNavigation();

    // =========================================================================
    // Bulk Selection for History Page
    // =========================================================================
    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl)
    })

    // Per page selector
    document.addEventListener('DOMContentLoaded', function() {
        const perPageSelect = document.getElementById('perPageSelect');
        if (perPageSelect) {
            perPageSelect.addEventListener('change', function() {
                const perPage = this.value;
                const url = new URL(window.location.href);
                url.searchParams.set('per_page', perPage);
                url.searchParams.set('page', '1');
                window.location.href = url.toString();
            });
        }
    })

    // Filtering and sorting functionality
    document.addEventListener('DOMContentLoaded', function() {
        const searchInput = document.getElementById('searchInput');
        const statusFilter = document.getElementById('statusFilter');
        const sortOrder = document.getElementById('sortOrder');
        const table = document.getElementById('sessionsTable');
        const tbody = table.querySelector('tbody');

        // Store original rows
        const originalRows = Array.from(tbody.querySelectorAll('tr'));

        function filterAndSort() {
            const searchTerm = searchInput.value.toLowerCase();
            const statusValue = statusFilter.value.toLowerCase();
            const sortValue = sortOrder.value;

            // Filter rows
            let filteredRows = originalRows.filter(row => {
                const filename = row.cells[2].textContent.toLowerCase();
                const status = row.querySelector('.badge').textContent.toLowerCase();
                const date = row.cells[1].textContent.toLowerCase();

                const matchesSearch = !searchTerm ||
                    filename.includes(searchTerm) ||
                    date.includes(searchTerm);

                const matchesStatus = !statusValue || status.includes(statusValue);

                return matchesSearch && matchesStatus;
            });

            // Sort rows
            filteredRows.sort((a, b) => {
                switch(sortValue) {
                    case 'oldest':
                        // Extract date from the strong tag
                        const dateA = a.cells[1].querySelector('strong').textContent;
                        const dateB = b.cells[1].querySelector('strong').textContent;
                        return new Date(dateA) - new Date(dateB);
                    case 'filename':
                        return a.cells[2].textContent.localeCompare(b.cells[2].textContent);
                    case 'newest':
                    default:
                        // Extract date from the strong tag
                        const dateA_newest = a.cells[1].querySelector('strong').textContent;
                        const dateB_newest = b.cells[1].querySelector('strong').textContent;
                        return new Date(dateB_newest) - new Date(dateA_newest);
                }
            });

            // Clear and repopulate tbody
            tbody.innerHTML = '';
            filteredRows.forEach(row => tbody.appendChild(row));

            // Show "no results" message if needed
            if (filteredRows.length === 0) {
                const noResultsRow = document.createElement('tr');
                noResultsRow.innerHTML = `
                    <td colspan="9" class="text-center text-muted py-4">
                        <em>No sessions match your current filters</em>
                    </td>
                `;
                tbody.appendChild(noResultsRow);
            }
        }

        // Add event listeners
        searchInput.addEventListener('input', filterAndSort);
        statusFilter.addEventListener('change', filterAndSort);
        sortOrder.addEventListener('change', filterAndSort);
    });

    // Handle delete session functionality
    document.addEventListener('DOMContentLoaded', function() {
        const deleteModal = document.getElementById('deleteModal');
        const sessionToDeleteElement = document.getElementById('sessionToDelete');
        const confirmDeleteButton = document.getElementById('confirmDelete');
        let sessionIdToDelete = null;

        // Handle delete button clicks
        document.addEventListener('click', function(e) {
            if (e.target.closest('[data-bs-target="#deleteModal"]')) {
                const button = e.target.closest('[data-bs-target="#deleteModal"]');
                sessionIdToDelete = button.getAttribute('data-session-id');
                const sessionName = button.getAttribute('data-session-name');
                sessionToDeleteElement.textContent = sessionName;

                // Check if this is a processing session
                const row = button.closest('tr');
                const statusBadge = row.querySelector('.badge');
                const isProcessing = statusBadge && statusBadge.textContent.trim() === 'Processing';

                // Update modal content based on session status
                const deleteDescription = document.getElementById('deleteDescription');
                const modalTitle = document.getElementById('deleteModalLabel');

                if (isProcessing) {
                    modalTitle.textContent = 'Cancel Processing';
                    deleteDescription.textContent = 'This will cancel the current processing and remove the session. This action cannot be undone.';
                } else {
                    modalTitle.textContent = 'Confirm Deletion';
                    deleteDescription.textContent = 'This action cannot be undone. All related data (conduits, nodes, subcatchments) will be permanently deleted.';
                }

                // Reset modal state
                const deleteText = confirmDeleteButton.querySelector('.delete-text');
                const deleteLoading = confirmDeleteButton.querySelector('.delete-loading');
                const cancelButton = deleteModal.querySelector('[data-bs-dismiss="modal"]');

                const buttonText = isProcessing ? 'Cancel Processing' : 'Delete Session';
                deleteText.innerHTML = `<i class="fas fa-trash me-1" aria-hidden="true"></i>${buttonText}`;
                deleteText.classList.remove('d-none');
                deleteLoading.classList.add('d-none');
                confirmDeleteButton.disabled = false;
                cancelButton.disabled = false;
                confirmDeleteButton.classList.remove('btn-success');
                confirmDeleteButton.classList.add('btn-outline-danger');
            }
        });

        // Handle confirm delete
        confirmDeleteButton.addEventListener('click', function() {
            if (sessionIdToDelete) {
                // Show loading state
                const deleteText = confirmDeleteButton.querySelector('.delete-text');
                const deleteLoading = confirmDeleteButton.querySelector('.delete-loading');
                const cancelButton = deleteModal.querySelector('[data-bs-dismiss="modal"]');

                deleteText.classList.add('d-none');
                deleteLoading.classList.remove('d-none');
                confirmDeleteButton.disabled = true;
                cancelButton.disabled = true;

                const deleteUrl = `/session/${sessionIdToDelete}/delete/`;
                const csrfToken = document.querySelector('[name=csrfmiddlewaretoken]')?.value;
                console.log('Delete URL:', deleteUrl);
                console.log('CSRF Token:', csrfToken);

                fetch(deleteUrl, {
                    method: 'POST',
                    headers: {
                        'X-CSRFToken': csrfToken,
                        'Content-Type': 'application/json'
                    }
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        // Show success state briefly before redirect
                        deleteLoading.classList.add('d-none');

                        // Check if this was a processing session for different success message
                        const modalTitle = document.getElementById('deleteModalLabel');
                        const isProcessing = modalTitle.textContent === 'Cancel Processing';
                        const successText = isProcessing ? 'Cancelled' : 'Deleted';

                        deleteText.innerHTML = `<i class="fas fa-check me-1"></i>${successText}`;
                        deleteText.classList.remove('d-none');
                        confirmDeleteButton.classList.remove('btn-outline-danger');
                        confirmDeleteButton.classList.add('btn-success');

                        // Redirect after short delay
                        setTimeout(() => {
                            window.location.reload();
                        }, 800);
                    } else {
                        // Reset loading state on error
                        deleteLoading.classList.add('d-none');
                        deleteText.classList.remove('d-none');
                        confirmDeleteButton.disabled = false;
                        cancelButton.disabled = false;
                        alert('Error: ' + data.message);
                    }
                })
                .catch(error => {
                    // Reset loading state on error
                    deleteLoading.classList.add('d-none');
                    deleteText.classList.remove('d-none');
                    confirmDeleteButton.disabled = false;
                    cancelButton.disabled = false;
                    console.error('Error:', error);
                    alert('An error occurred while deleting the session.');
                });
            }
        });
    });


    function initBulkSelection() {
        const bulkActionBar = document.getElementById('bulkActionBar');
        const selectedCountElement = document.getElementById('selectedCount');
        const selectAllCheckbox = document.getElementById('selectAllCheckbox');
        const selectAllVisibleBtn = document.getElementById('selectAllVisible');
        const clearSelectionBtn = document.getElementById('clearSelection');
        const bulkDeleteBtn = document.getElementById('bulkDeleteBtn');
        const bulkDeleteModalElement = document.getElementById('bulkDeleteModal');
        const confirmBulkDeleteBtn = document.getElementById('confirmBulkDelete');
        const bulkDeleteCountElement = document.getElementById('bulkDeleteCount');

        // Only initialize if bulk selection elements exist
        if (!bulkActionBar || !selectAllCheckbox) return;

        const bulkDeleteModal = new bootstrap.Modal(bulkDeleteModalElement);
        let selectedSessions = new Set();

        function updateUI() {
            const count = selectedSessions.size;
            selectedCountElement.textContent = count;
            bulkDeleteCountElement.textContent = count;

            // Show/hide bulk action bar
            if (count > 0) {
                bulkActionBar.style.display = 'block';
            } else {
                bulkActionBar.style.display = 'none';
            }

            // Update select all checkbox state
            const visibleRows = document.querySelectorAll('#sessionsTable tbody tr[data-session-id]');
            const visibleChecked = Array.from(visibleRows).filter(row => {
                const checkbox = row.querySelector('.row-checkbox');
                return checkbox && checkbox.checked;
            });

            if (visibleRows.length > 0 && visibleChecked.length === visibleRows.length) {
                selectAllCheckbox.checked = true;
                selectAllCheckbox.indeterminate = false;
            } else if (visibleChecked.length > 0) {
                selectAllCheckbox.checked = false;
                selectAllCheckbox.indeterminate = true;
            } else {
                selectAllCheckbox.checked = false;
                selectAllCheckbox.indeterminate = false;
            }

            // Update row styling
            document.querySelectorAll('#sessionsTable tbody tr[data-session-id]').forEach(row => {
                const sessionId = row.dataset.sessionId;
                if (selectedSessions.has(sessionId)) {
                    row.classList.add('selected');
                } else {
                    row.classList.remove('selected');
                }
            });
        }

        // Handle individual checkbox changes
        document.addEventListener('change', function(e) {
            if (e.target.classList.contains('row-checkbox')) {
                const sessionId = e.target.dataset.sessionId;
                if (e.target.checked) {
                    selectedSessions.add(sessionId);
                } else {
                    selectedSessions.delete(sessionId);
                }
                updateUI();
            }
        });

        // Handle select all checkbox
        selectAllCheckbox.addEventListener('change', function() {
            const visibleRows = document.querySelectorAll('#sessionsTable tbody tr[data-session-id]');
            visibleRows.forEach(row => {
                const checkbox = row.querySelector('.row-checkbox');
                const sessionId = row.dataset.sessionId;
                if (checkbox) {
                    checkbox.checked = this.checked;
                    if (this.checked) {
                        selectedSessions.add(sessionId);
                    } else {
                        selectedSessions.delete(sessionId);
                    }
                }
            });
            updateUI();
        });

        // Handle "Select all visible" button
        selectAllVisibleBtn.addEventListener('click', function() {
            const visibleRows = document.querySelectorAll('#sessionsTable tbody tr[data-session-id]');
            visibleRows.forEach(row => {
                const checkbox = row.querySelector('.row-checkbox');
                const sessionId = row.dataset.sessionId;
                if (checkbox) {
                    checkbox.checked = true;
                    selectedSessions.add(sessionId);
                }
            });
            updateUI();
        });

        // Handle "Clear selection" button
        clearSelectionBtn.addEventListener('click', function() {
            document.querySelectorAll('.row-checkbox').forEach(checkbox => {
                checkbox.checked = false;
            });
            selectAllCheckbox.checked = false;
            selectedSessions.clear();
            updateUI();
        });

        // Handle bulk delete button
        bulkDeleteBtn.addEventListener('click', function() {
            if (selectedSessions.size > 0) {
                bulkDeleteModal.show();
            }
        });

        // Handle confirm bulk delete
        confirmBulkDeleteBtn.addEventListener('click', function() {
            if (selectedSessions.size === 0) return;

            const bulkDeleteText = confirmBulkDeleteBtn.querySelector('.bulk-delete-text');
            const bulkDeleteLoading = confirmBulkDeleteBtn.querySelector('.bulk-delete-loading');
            const cancelButton = document.querySelector('#bulkDeleteModal [data-bs-dismiss="modal"]');

            // Show loading state
            bulkDeleteText.classList.add('d-none');
            bulkDeleteLoading.classList.remove('d-none');
            confirmBulkDeleteBtn.disabled = true;
            cancelButton.disabled = true;

            const csrfToken = document.querySelector('[name=csrfmiddlewaretoken]')?.value;
            const sessionIds = Array.from(selectedSessions).map(id => parseInt(id));

            fetch('/sessions/bulk-delete/', {
                method: 'POST',
                headers: {
                    'X-CSRFToken': csrfToken,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ session_ids: sessionIds })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // Show success state briefly before reload
                    bulkDeleteLoading.classList.add('d-none');
                    bulkDeleteText.innerHTML = `<i class="fas fa-check me-1"></i>Deleted ${data.deleted_count} session(s)`;
                    bulkDeleteText.classList.remove('d-none');
                    confirmBulkDeleteBtn.classList.remove('btn-danger');
                    confirmBulkDeleteBtn.classList.add('btn-success');

                    setTimeout(() => {
                        window.location.reload();
                    }, 800);
                } else {
                    // Reset loading state on error
                    bulkDeleteLoading.classList.add('d-none');
                    bulkDeleteText.classList.remove('d-none');
                    confirmBulkDeleteBtn.disabled = false;
                    cancelButton.disabled = false;
                    alert('Error: ' + data.message);
                }
            })
            .catch(error => {
                // Reset loading state on error
                bulkDeleteLoading.classList.add('d-none');
                bulkDeleteText.classList.remove('d-none');
                confirmBulkDeleteBtn.disabled = false;
                cancelButton.disabled = false;
                console.error('Error:', error);
                alert('An error occurred while deleting sessions.');
            });
        });

        // Reset bulk delete modal state when hidden
        bulkDeleteModalElement.addEventListener('hidden.bs.modal', function() {
            const bulkDeleteText = confirmBulkDeleteBtn.querySelector('.bulk-delete-text');
            const bulkDeleteLoading = confirmBulkDeleteBtn.querySelector('.bulk-delete-loading');

            bulkDeleteText.innerHTML = '<i class="fas fa-trash me-1" aria-hidden="true"></i>Delete All Selected';
            bulkDeleteText.classList.remove('d-none');
            bulkDeleteLoading.classList.add('d-none');
            confirmBulkDeleteBtn.disabled = false;
            confirmBulkDeleteBtn.classList.remove('btn-success');
            if (!confirmBulkDeleteBtn.classList.contains('btn-danger')) {
                confirmBulkDeleteBtn.classList.add('btn-danger');
            }
        });
    }

    initBulkSelection();

});
