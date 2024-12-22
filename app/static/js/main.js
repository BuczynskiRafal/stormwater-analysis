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
        initComplete: function () {
            const label = $(table).parents('.dataTables_wrapper').find('label');
            let labelHtml = label.html();
            if (labelHtml) {
                label.html(labelHtml.replace('Search:', 'Search: '));
            }
        }
    });
}
