export interface KeyValue {
	key: string
	value: string
}

export interface MetadataItem {
	id: string
	attrs: KeyValue[]
}

export interface GraphNodeStyle {
	backgroundColor?: string
	borderColor?: string
	textColor?: string
	borderWidth?: number
}

export interface GraphNode {
	id: string
	label: string
	namespace: string
	attrs: KeyValue[]
	outputsMetadata?: MetadataItem[]
	inputsMetadata?: MetadataItem[]
	incomingEdges: Array<{
		sourceNodeId: string
		sourceNodeOutputId?: string
		targetNodeInputId?: string
	}>
	subgraphIds?: string[]
	style?: GraphNodeStyle
}

export interface EdgeData {
	sourceNodeId: string
	targetNodeId: string
	label?: string
}

export interface EdgeOverlay {
	name: string
	edges: EdgeData[]
	edgeColor?: string
	edgeWidth?: number
	edgeLabelFontSize?: number
}

export interface EdgeOverlaysData {
	type: 'EDGE_OVERLAYS'
	name: string
	overlays: EdgeOverlay[]
}

export interface TasksData {
	edgeOverlaysDataListLeftPane?: EdgeOverlaysData[]
	edgeOverlaysDataListRightPane?: EdgeOverlaysData[]
}

export interface ModelExplorerGraph {
	id: string
	nodes: GraphNode[]
	tasksData?: TasksData
}

export interface ModelExplorerGraphs {
	graphs: ModelExplorerGraph[]
	_metadata?: Record<string, unknown>
}
