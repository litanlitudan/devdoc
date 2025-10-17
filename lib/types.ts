import { Server as HttpServer } from 'node:http'
import { WebSocketServer } from 'ws'
import { FSWatcher } from 'chokidar'

export interface Flags {
	port: string | number
	livereloadport: number | 'false'
	address: string
	silent: boolean
	verbose: boolean
	watch?: boolean
	browser?: boolean | 'false'
	dir: string
	$pathProvided?: boolean
	$openLocation?: string | boolean
	[key: string]: any
}

export interface DevdocService {
	pid: number | undefined
	httpServer: HttpServer
	liveReloadServer?: {
		wss: WebSocketServer | null
		watcher: FSWatcher | null
		port: number
	}
	expressApp: HttpServer
	launchUrl: string | false
}

export interface FileInfo {
	name: string
	fullPath: string
	isDirectory: boolean
	mtime: Date
	birthtime: Date
	size: number
}

export interface DirectoryInfo {
	html: string
	fileCount: number
	folderCount: number
}

export interface Breadcrumb {
	href: string
	text: string
}

export interface ImplantOptions {
	maxDepth: number
	baseDir?: string
}

export interface ImplantHandlers {
	[key: string]: (
		url: string,
		opts?: ImplantOptions,
	) => Promise<string | false>
}

export interface FileTypes {
	markdown: string[]
	html: string[]
	mlir: string[]
	onnx: string[]
	diff: string[]
	text: string[]
	watch: string[]
	exclusions: string[]
}

export interface MaterialIcons {
	folderNames: { [key: string]: string }
	fileExtensions: { [key: string]: string }
	fileNames: { [key: string]: string }
}

export interface HttpServerResult {
	server: HttpServer
	port: number
}
