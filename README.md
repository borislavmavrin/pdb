# pdb

HOG2 in XCode:
1. clone from hog2-PDB-refactor branch
2. in XCode open hog2-PDB-refactor/build/XCode/hog2 glut/hog2 glut.xcodeproj
3. in project settings->Build Phases remove PancakePuzzle.cpp from "Compile Sourses"
4. pick the target "Sliding Tile Puzzle"
5. modify apps/stp/Driver.cpp (I uploaded my version to the root of this repo)
