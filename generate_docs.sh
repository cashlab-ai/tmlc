#!/bin/bash

pip install -r requirements.txt

package_name=tmlc
docs_folder=docs

# Generate UML diagrams
pyreverse -o svg -p $package_name $package_name
mv ${package_name}_classes.svg $docs_folder/
mv ${package_name}_packages.svg $docs_folder/

# Add diagrams to the index.md file
cat <<EOT >> $docs_folder/index.md
## Class Diagram
![Class Diagram](${package_name}_classes.svg)
## Package Diagram
![Package Diagram](${package_name}_packages.svg)
EOT

# Build and serve the documentation
mkdocs serve
