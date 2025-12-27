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

                                // Calculate clipping to stay within accordion bounds
                                const accordionPadding = 15;
                                const accordionRightEdge = accordionItemRect.right - accordionPadding;
                                const maxWidth = accordionRightEdge - containerRect.left;


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

                                        // Handle scrolling: first column stays frozen, others scroll
                                        if (index === 0) {
                                            // First column - position exactly like original
                                            clonedTh.style.position = 'sticky';
                                            clonedTh.style.left = '0px';
                                            clonedTh.style.zIndex = '1019';
                                        } else {
                                            // Other columns - apply scroll offset but maintain exact positioning
                                            clonedTh.style.transform = `translateX(-${Math.round(scrollLeft)}px)`;
                                        }

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
                                    clonedHeaders.forEach((clonedTh, index) => {
                                        if (index > 0) { // Skip first column
                                            clonedTh.style.transform = `translateX(-${scrollLeft}px)`;
                                        }
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
                                        clonedHeaders.forEach((clonedTh, index) => {
                                            if (index > 0) {
                                                clonedTh.style.transform = `translateX(-${newScrollLeft}px)`;
                                            }
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

                                    const accordionPadding = 15;
                                    const accordionRightEdge = accordionItemRect.right - accordionPadding;
                                    const maxWidth = accordionRightEdge - containerRect.left;

                                    wrapperDiv.style.top = (navBottomPosition + accordionHeaderHeight) + 'px';
                                    wrapperDiv.style.left = containerRect.left + 'px';
                                    wrapperDiv.style.width = Math.min(containerRect.width, maxWidth) + 'px';
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

});
