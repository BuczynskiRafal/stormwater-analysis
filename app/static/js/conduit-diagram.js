/**
 * ConduitDiagram - Technical diagram generator for conduit visualization
 * Uses SVG.js library for rendering
 *
 * Features:
 * - Draws manholes (inlet/outlet) with covers
 * - Draws connecting pipe with slope visualization
 * - Shows water filling level
 * - Displays dimension table with elevations
 * - Supports zoom and pan interactions
 */
class ConduitDiagram {
    constructor(containerId, data) {
        this.container = document.getElementById(containerId);
        this.data = data;
        this.draw = null;
        this.svgElement = null;

        // Zoom/pan state
        this.scale = 1;
        this.translateX = 0;
        this.translateY = 0;
        this.initialScale = 1;
        this.initialTranslateX = 0;
        this.initialTranslateY = 0;

        // Drag state
        this.isDragging = false;
        this.lastMouseX = 0;
        this.lastMouseY = 0;

        // Drawing constants
        this.MARGIN = 50;
        this.MANHOLE_WIDTH = 30;
        this.MANHOLE_HEIGHT = 80;
        this.PIPE_HEIGHT = 20;
        this.SLOPE_EXAGGERATION = 200;

        // Colors
        this.COLORS = {
            ground: '#228B22',
            manhole: '#000000',
            cover: '#696969',
            water: '#d5e3f5',
            waterLine: 'var(--gray-500)',
            text: '#7a7a7a',
            axis: '#000000'
        };

        // Typography
        this.FONT = {
            size: 12,
            numberSize: 10,
            titleSize: 20
        };
    }

    /**
     * Initialize the diagram
     */
    init() {
        if (!this.container) {
            console.error('ConduitDiagram: Container not found');
            return;
        }

        if (typeof SVG === 'undefined') {
            console.error('ConduitDiagram: SVG.js library not loaded');
            return;
        }

        const containerWidth = this.container.offsetWidth;
        const containerHeight = this.container.offsetHeight;

        if (containerWidth === 0 || containerHeight === 0) {
            console.error('ConduitDiagram: Container has no dimensions');
            return;
        }

        this.containerWidth = containerWidth;
        this.containerHeight = containerHeight;

        // Create SVG canvas
        this.draw = SVG().addTo(this.container).size(containerWidth, containerHeight);
        this.svgElement = this.draw.node;

        this.setupZoomPan();
        this.render();
        this.centerView();
        this.setupResetButton();
    }

    /**
     * Setup zoom and pan interactions
     */
    setupZoomPan() {
        // Mouse wheel zoom
        this.svgElement.addEventListener('wheel', (e) => {
            e.preventDefault();
            const rect = this.svgElement.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;

            const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
            const newScale = Math.max(0.5, Math.min(5, this.scale * zoomFactor));

            if (newScale !== this.scale) {
                const scaleChange = newScale / this.scale;
                this.translateX += (mouseX / this.scale - mouseX / newScale);
                this.translateY += (mouseY / this.scale - mouseY / newScale);
                this.scale = newScale;
                this.updateTransform();
            }
        });

        // Mouse drag pan
        this.svgElement.addEventListener('mousedown', (e) => {
            this.isDragging = true;
            this.lastMouseX = e.clientX;
            this.lastMouseY = e.clientY;
            this.container.style.cursor = 'grabbing';
            e.preventDefault();
        });

        this.svgElement.addEventListener('mousemove', (e) => {
            if (this.isDragging) {
                const deltaX = (e.clientX - this.lastMouseX) / this.scale;
                const deltaY = (e.clientY - this.lastMouseY) / this.scale;
                this.translateX -= deltaX;
                this.translateY -= deltaY;
                this.lastMouseX = e.clientX;
                this.lastMouseY = e.clientY;
                this.updateTransform();
            }
        });

        this.svgElement.addEventListener('mouseup', () => {
            this.isDragging = false;
            this.container.style.cursor = 'grab';
        });

        this.svgElement.addEventListener('mouseleave', () => {
            this.isDragging = false;
            this.container.style.cursor = 'grab';
        });
    }

    /**
     * Update SVG viewBox based on current transform
     */
    updateTransform() {
        this.draw.viewbox(
            this.translateX,
            this.translateY,
            this.containerWidth / this.scale,
            this.containerHeight / this.scale
        );
    }

    /**
     * Setup reset zoom button
     */
    setupResetButton() {
        const resetButton = document.getElementById('reset-zoom');
        if (resetButton) {
            resetButton.addEventListener('click', () => {
                this.resetView();
                resetButton.blur();
            });
        }
    }

    /**
     * Reset view to initial state
     */
    resetView() {
        this.scale = this.initialScale;
        this.translateX = this.initialTranslateX;
        this.translateY = this.initialTranslateY;
        this.updateTransform();
    }

    /**
     * Main render method - draws all diagram elements
     */
    render() {
        this.calculatePositions();
        this.drawGroundLine();
        this.drawInletManhole();
        this.drawOutletManhole();
        this.drawPipe();
        this.drawWaterInManholes();
        this.drawConduitName();
        this.drawNodeLabels();
        this.drawDimensionTable();
        this.drawFillingIndicator();
    }

    /**
     * Calculate all positions for drawing
     */
    calculatePositions() {
        const pipeLength = this.containerWidth - 2 * this.MARGIN - 100;
        const centerY = this.containerHeight / 2;

        // Calculate elevation difference for slope visualization
        const elevationDiff = (this.data.inletElevation - this.data.outletElevation) * this.SLOPE_EXAGGERATION;

        // Ground positions
        this.inletGroundY = centerY - this.MANHOLE_HEIGHT / 2 - this.data.inletCover * 10 - 20;
        this.outletGroundY = centerY - this.MANHOLE_HEIGHT / 2 - this.data.outletCover * 10 - 20 + elevationDiff;

        // Pipe bottom positions
        this.inletPipeBottomY = centerY + this.PIPE_HEIGHT / 2;
        this.outletPipeBottomY = centerY + this.PIPE_HEIGHT / 2 + elevationDiff;

        // Manhole positions
        this.inletX = this.MARGIN;
        this.inletY = this.inletPipeBottomY;
        this.outletX = this.containerWidth - this.MARGIN - this.MANHOLE_WIDTH;
        this.outletY = this.outletPipeBottomY;

        // Axis centers
        this.inletAxisX = this.inletX + this.MANHOLE_WIDTH / 2;
        this.outletAxisX = this.outletX + this.MANHOLE_WIDTH / 2;

        // Pipe positions
        this.pipeStartX = this.inletX + this.MANHOLE_WIDTH;
        this.pipeStartY = this.inletPipeBottomY - this.PIPE_HEIGHT;
        this.pipeEndX = this.outletX;
        this.pipeEndY = this.outletPipeBottomY - this.PIPE_HEIGHT;

        // Water level
        this.fillingHeightVisual = (this.data.filling / this.data.diameter) * this.PIPE_HEIGHT;
    }

    /**
     * Draw continuous ground line
     */
    drawGroundLine() {
        this.draw.line(this.inletAxisX, this.inletGroundY, this.outletAxisX, this.outletGroundY)
            .stroke({ color: this.COLORS.ground, width: 1 });
    }

    /**
     * Draw inlet manhole structure
     */
    drawInletManhole() {
        const cylinderHeight = Math.abs(this.inletY - this.inletGroundY);

        // Manhole structure
        this.draw.rect(this.MANHOLE_WIDTH, cylinderHeight)
            .move(this.inletX, Math.min(this.inletY, this.inletGroundY))
            .fill('none')
            .stroke({ color: this.COLORS.manhole, width: 1 });

        // Manhole cover
        this.draw.rect(this.MANHOLE_WIDTH + 10, 8)
            .move(this.inletX - 5, this.inletGroundY - 4)
            .fill(this.COLORS.cover)
            .stroke({ color: this.COLORS.manhole, width: 1 });

        // Axis line
        const bottom = Math.max(this.inletY, this.inletGroundY);
        const coverTop = this.inletGroundY - 4 + 8;
        this.draw.line(this.inletAxisX, bottom, this.inletAxisX, coverTop)
            .stroke({ color: this.COLORS.axis, width: 0.5 })
            .attr('stroke-dasharray', '6,2,1,2');

        // Store cover top for labels
        this.inletCoverTop = coverTop;
    }

    /**
     * Draw outlet manhole structure
     */
    drawOutletManhole() {
        const cylinderHeight = Math.abs(this.outletY - this.outletGroundY);

        // Manhole structure
        this.draw.rect(this.MANHOLE_WIDTH, cylinderHeight)
            .move(this.outletX, Math.min(this.outletY, this.outletGroundY))
            .fill('none')
            .stroke({ color: this.COLORS.manhole, width: 1 });

        // Manhole cover
        this.draw.rect(this.MANHOLE_WIDTH + 10, 8)
            .move(this.outletX - 5, this.outletGroundY - 4)
            .fill(this.COLORS.cover)
            .stroke({ color: this.COLORS.manhole, width: 1 });

        // Axis line
        const bottom = Math.max(this.outletY, this.outletGroundY);
        const coverTop = this.outletGroundY - 4 + 8;
        this.draw.line(this.outletAxisX, bottom, this.outletAxisX, coverTop)
            .stroke({ color: this.COLORS.axis, width: 0.5 })
            .attr('stroke-dasharray', '6,2,1,2');

        // Store cover top for labels
        this.outletCoverTop = coverTop;
    }

    /**
     * Draw the connecting pipe
     */
    drawPipe() {
        // Pipe outline
        this.draw.polygon([
            [this.pipeStartX, this.pipeStartY],
            [this.pipeEndX, this.pipeEndY],
            [this.pipeEndX, this.pipeEndY + this.PIPE_HEIGHT],
            [this.pipeStartX, this.pipeStartY + this.PIPE_HEIGHT]
        ]).fill('none').stroke({ color: this.COLORS.manhole, width: 1 });

        // Water level positions
        const waterLevelStartY = this.pipeStartY + this.PIPE_HEIGHT - this.fillingHeightVisual;
        const waterLevelEndY = this.pipeEndY + this.PIPE_HEIGHT - this.fillingHeightVisual;

        // Water filling
        this.draw.polygon([
            [this.pipeStartX, waterLevelStartY],
            [this.pipeEndX, waterLevelEndY],
            [this.pipeEndX, this.pipeEndY + this.PIPE_HEIGHT],
            [this.pipeStartX, this.pipeStartY + this.PIPE_HEIGHT]
        ]).fill(this.COLORS.water).opacity(0.4).stroke({ width: 0 });

        // Water level line
        this.draw.line(this.pipeStartX, waterLevelStartY, this.pipeEndX, waterLevelEndY)
            .stroke({ color: this.COLORS.waterLine, width: 0.5 });

        // Store for filling indicator
        this.waterLevelMidY = (waterLevelStartY + waterLevelEndY) / 2;
    }

    /**
     * Draw water in manholes
     */
    drawWaterInManholes() {
        const inletWaterLevel = this.inletPipeBottomY - this.fillingHeightVisual;
        const outletWaterLevel = this.outletPipeBottomY - this.fillingHeightVisual;

        // Inlet manhole water
        const inletManholeBottom = Math.max(this.inletY, this.inletGroundY);
        if (inletWaterLevel < inletManholeBottom) {
            this.draw.rect(this.MANHOLE_WIDTH, inletManholeBottom - inletWaterLevel)
                .move(this.inletX, inletWaterLevel)
                .fill(this.COLORS.water)
                .opacity(0.4)
                .stroke({ width: 0 });

            this.draw.line(this.inletX, inletWaterLevel, this.inletX + this.MANHOLE_WIDTH, inletWaterLevel)
                .stroke({ color: this.COLORS.waterLine, width: 0.5 });
        }

        // Outlet manhole water
        const outletManholeBottom = Math.max(this.outletY, this.outletGroundY);
        if (outletWaterLevel < outletManholeBottom) {
            this.draw.rect(this.MANHOLE_WIDTH, outletManholeBottom - outletWaterLevel)
                .move(this.outletX, outletWaterLevel)
                .fill(this.COLORS.water)
                .opacity(0.4)
                .stroke({ width: 0 });

            this.draw.line(this.outletX, outletWaterLevel, this.outletX + this.MANHOLE_WIDTH, outletWaterLevel)
                .stroke({ color: this.COLORS.waterLine, width: 0.5 });
        }
    }

    /**
     * Draw conduit name title
     */
    drawConduitName() {
        this.draw.text(this.data.name)
            .move(this.containerWidth / 2, 80)
            .fill(this.COLORS.text)
            .font({ size: this.FONT.titleSize, weight: 'bold' });
    }

    /**
     * Draw node labels above manholes
     */
    drawNodeLabels() {
        const labelOffset = 30;

        this.draw.text(this.data.inletNode)
            .move(this.inletAxisX, this.inletCoverTop - labelOffset)
            .fill(this.COLORS.text)
            .font({ size: this.FONT.size })
            .attr('text-anchor', 'middle');

        this.draw.text(this.data.outletNode)
            .move(this.outletAxisX, this.outletCoverTop - labelOffset)
            .fill(this.COLORS.text)
            .font({ size: this.FONT.size })
            .attr('text-anchor', 'middle');
    }

    /**
     * Draw filling level indicator with triangle
     */
    drawFillingIndicator() {
        const pipeLength = this.pipeEndX - this.pipeStartX;
        const diameterX = this.pipeStartX + pipeLength / 2;
        const triangleSize = 10;
        const triangleX = diameterX - 15;

        // Triangle indicator
        this.draw.polygon([
            [triangleX - triangleSize / 2, this.waterLevelMidY - triangleSize],
            [triangleX + triangleSize / 2, this.waterLevelMidY - triangleSize],
            [triangleX, this.waterLevelMidY]
        ]).fill('none').stroke({ color: this.COLORS.text, width: 0.8 });

        // Horizontal line from triangle
        this.draw.line(triangleX, this.waterLevelMidY, triangleX + 10, this.waterLevelMidY)
            .stroke({ color: this.COLORS.text, width: 0.8 });

        // Filling label
        const fillingPercentage = (this.data.filling / this.data.diameter) * 100;
        this.draw.text(`${this.data.filling.toFixed(2)}m (${fillingPercentage.toFixed(1)}%)`)
            .move(diameterX - 10, this.waterLevelMidY - 8 - triangleSize - 2)
            .fill(this.COLORS.text)
            .font({ size: this.FONT.size });
    }

    /**
     * Draw dimension table below the diagram
     */
    drawDimensionTable() {
        const lineHeight = 36;
        const labelTableWidth = 160;

        // Calculate table position
        const lowerManholeY = Math.max(this.inletY, this.outletY);
        const tableY = lowerManholeY + this.MANHOLE_HEIGHT - 50;

        const tableStartX = this.inletAxisX;
        const tableEndX = this.outletAxisX;
        const labelX = this.inletX - 10;
        const labelTableStartX = labelX - labelTableWidth;

        let currentY = tableY;

        // Top border line
        this.draw.line(labelTableStartX, tableY - 2, this.inletAxisX, tableY - 2)
            .stroke({ color: '#000', width: 0.5 });

        // Row 1: Ground Elevation
        this.drawTableRow(
            labelTableStartX, tableStartX, tableEndX, currentY, lineHeight,
            'Ground Elevation',
            this.data.inletElevation.toFixed(2),
            this.data.outletElevation.toFixed(2)
        );
        currentY += lineHeight;

        // Row 2: Invert Elevation
        const inletBottomElev = (this.data.inletElevation - this.data.inletCover).toFixed(2);
        const outletBottomElev = (this.data.outletElevation - this.data.outletCover).toFixed(2);
        this.drawTableRow(
            labelTableStartX, tableStartX, tableEndX, currentY, lineHeight,
            'Invert Elevation',
            inletBottomElev,
            outletBottomElev
        );
        currentY += lineHeight;

        // Row 3: Cover Depth
        this.drawTableRow(
            labelTableStartX, tableStartX, tableEndX, currentY, lineHeight,
            'Cover Depth',
            this.data.inletCover.toFixed(2),
            this.data.outletCover.toFixed(2)
        );
        currentY += lineHeight;

        // Row 4: Channel Parameters
        this.drawChannelParamsRow(
            labelTableStartX, tableStartX, tableEndX, currentY, lineHeight
        );

        // Vertical borders
        this.draw.line(tableStartX, tableY - 2, tableStartX, currentY + lineHeight - 2)
            .stroke({ color: '#000', width: 0.5 });
        this.draw.line(tableEndX, tableY - 2, tableEndX, currentY + lineHeight - 2)
            .stroke({ color: '#000', width: 0.5 });
        this.draw.line(labelTableStartX, tableY - 2, labelTableStartX, currentY + lineHeight - 2)
            .stroke({ color: '#000', width: 0.5 });
        this.draw.line(labelX, tableY - 2, labelX, currentY + lineHeight - 2)
            .stroke({ color: '#000', width: 0.5 });

        // Connection lines from table to manhole bottoms
        const inletBottom = Math.max(this.inletY, this.inletGroundY);
        const outletBottom = Math.max(this.outletY, this.outletGroundY);
        this.draw.line(this.inletAxisX, tableY - 2, this.inletAxisX, inletBottom)
            .stroke({ color: '#000', width: 0.5 });
        this.draw.line(this.outletAxisX, tableY - 2, this.outletAxisX, outletBottom)
            .stroke({ color: '#000', width: 0.5 });

        // Store table bounds for centering
        this.tableEndY = currentY + lineHeight;
        this.tableLabelStartX = labelTableStartX;
    }

    /**
     * Draw a single table row
     */
    drawTableRow(labelStartX, tableStartX, tableEndX, y, lineHeight, label, inletValue, outletValue) {
        const labelX = this.inletX - 10;

        // Label
        this.draw.text(label)
            .move(labelStartX + 5, y + lineHeight / 2 - 6)
            .fill(this.COLORS.text)
            .font({ size: this.FONT.size })
            .attr('text-anchor', 'start');

        // Horizontal lines
        this.draw.line(tableStartX, y - 2, tableEndX, y - 2)
            .stroke({ color: '#000', width: 0.5 });
        this.draw.line(tableStartX, y + lineHeight - 2, tableEndX, y + lineHeight - 2)
            .stroke({ color: '#000', width: 0.5 });
        this.draw.line(labelStartX, y + lineHeight - 2, this.inletAxisX, y + lineHeight - 2)
            .stroke({ color: '#000', width: 0.5 });

        // Values (rotated)
        this.draw.text(inletValue)
            .move(this.inletAxisX - 20, y + lineHeight / 2 - 5)
            .fill(this.COLORS.text)
            .font({ size: this.FONT.numberSize })
            .transform({ rotate: 270 });

        this.draw.text(outletValue)
            .move(this.outletAxisX - 20, y + lineHeight / 2 - 5)
            .fill(this.COLORS.text)
            .font({ size: this.FONT.numberSize })
            .transform({ rotate: 270 });
    }

    /**
     * Draw channel parameters row
     */
    drawChannelParamsRow(labelStartX, tableStartX, tableEndX, y, lineHeight) {
        // Label
        this.draw.text('Diameter, Length, Slope')
            .move(labelStartX + 5, y + lineHeight / 2 - 6)
            .fill(this.COLORS.text)
            .font({ size: this.FONT.size })
            .attr('text-anchor', 'start');

        // Horizontal lines
        this.draw.line(tableStartX, y + lineHeight - 2, tableEndX, y + lineHeight - 2)
            .stroke({ color: '#000', width: 0.5 });
        this.draw.line(labelStartX, y + lineHeight - 2, this.inletAxisX, y + lineHeight - 2)
            .stroke({ color: '#000', width: 0.5 });

        // Parameters centered
        const channelParamsX = tableStartX + (tableEndX - tableStartX) / 2;
        const paramsText = `\u00d8 ${this.data.diameter.toFixed(3)} m, L = ${this.data.length.toFixed(1)} m, i = ${this.data.slope.toFixed(2)}\u2030`;
        this.draw.text(paramsText)
            .move(channelParamsX, y + lineHeight / 2 - 6)
            .fill(this.COLORS.text)
            .font({ size: this.FONT.size })
            .attr('text-anchor', 'middle');
    }

    /**
     * Center the view to show the entire diagram
     */
    centerView() {
        const nodeLabelOffset = 30;

        // Calculate bounds
        const minX = this.tableLabelStartX - 10;
        const maxX = this.containerWidth - 10;
        const minY = Math.min(this.inletCoverTop, this.outletCoverTop) - nodeLabelOffset - 20;
        const maxY = this.tableEndY + 10;

        const drawingWidth = maxX - minX;
        const drawingHeight = maxY - minY;

        // Calculate scale to fit with padding
        const scaleX = this.containerWidth / drawingWidth;
        const scaleY = this.containerHeight / drawingHeight;
        const calculatedScale = Math.min(scaleX, scaleY, 1) * 0.9;

        // Calculate center position
        const drawingCenterX = (minX + maxX) / 2;
        const drawingCenterY = (minY + maxY) / 2;
        const calculatedTranslateX = drawingCenterX - (this.containerWidth / calculatedScale) / 2;
        const calculatedTranslateY = drawingCenterY - (this.containerHeight / calculatedScale) / 2;

        // Store initial view settings
        this.initialScale = calculatedScale;
        this.initialTranslateX = calculatedTranslateX;
        this.initialTranslateY = calculatedTranslateY;

        // Apply initial view
        this.scale = this.initialScale;
        this.translateX = this.initialTranslateX;
        this.translateY = this.initialTranslateY;
        this.updateTransform();
    }
}

// Auto-initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    const container = document.getElementById('conduit-diagram');
    const dataElement = document.getElementById('conduit-data');

    if (container && dataElement) {
        // Small delay to ensure SVG.js is loaded
        setTimeout(() => {
            if (typeof SVG !== 'undefined') {
                try {
                    const data = JSON.parse(dataElement.textContent);
                    const diagram = new ConduitDiagram('conduit-diagram', data);
                    diagram.init();
                } catch (error) {
                    console.error('ConduitDiagram: Failed to parse data', error);
                }
            } else {
                console.error('ConduitDiagram: SVG.js not loaded');
            }
        }, 100);
    }
});
