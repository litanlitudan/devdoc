import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { dirname } from 'node:path'

const __filename = fileURLToPath(import.meta.url)
const __dirname = dirname(__filename)


const materialIcons = require(path.join(__dirname, 'icons', 'material-icons.json'))

const iconCSS = []

iconCSS.push(`.icon {
    content: "";
    display: block;
    position: absolute;
    width: 24px;
    height: 24px;
    float: left;
    margin: 0 -28px 0 0;
}
`)

Reflect.ownKeys(materialIcons.iconDefinitions).forEach(iconName => {
	const iconFile = materialIcons.iconDefinitions[iconName].iconPath
	iconCSS.push(`.icon.${iconName}:before {
    background: url("{markserv}icons/${iconFile}") no-repeat;
}
`)
})

fs.writeFileSync(path.join(__dirname, 'icons', 'icons.css'), iconCSS.join(''))
