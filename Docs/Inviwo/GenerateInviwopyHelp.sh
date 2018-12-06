#!/bin/bash

# Modify to your own paths for built inviwo and inviwo source code
# NOTE the document generator code from inviwo can be faulty and need to be modified
# Close inviwo after generating documents
~/inviwo_build/bin/inviwo -w ~/inviwo/data/workspaces/boron.inv -p ~/inviwo/modules/python3/scripts/documentgenerator.py

google-chrome ~/inviwo/data/help/inviwopy.html