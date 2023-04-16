#!/bin/bash

pip install -r docs/requirements.txt

package_name=tmlc
docs_folder=docs

# Generate UML diagrams
pyreverse -o svg -p $package_name $package_name
mv classes_${package_name}.svg $docs_folder/imgs
mv packages_${package_name}.svg $docs_folder/imgs

# Add diagrams to the index.md file
cat <<EOT > $docs_folder/uml.md
## Class Diagram
![Class Diagram](imgs/classes_${package_name}.svg)
## Package Diagram
![Package Diagram](imgs/packages_${package_name}.svg)
EOT

# Build and serve the documentation
mkdocs serve
