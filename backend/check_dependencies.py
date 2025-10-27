#!/usr/bin/env python3
"""
Check if all required dependencies are available
"""

def check_dependencies():
    """Check if all required dependencies are available"""
    dependencies = [
        "cv2",
        "numpy",
        "torch",
        "deepface",
        "ultralytics",
        "pickle",
        "os",
        "typing",
        "tensorflow",
        "sklearn",
        "joblib"
    ]
    
    missing = []
    
    for dep in dependencies:
        try:
            if dep == "cv2":
                import cv2
            elif dep == "numpy":
                import numpy
            elif dep == "torch":
                import torch
            elif dep == "deepface":
                import deepface
            elif dep == "ultralytics":
                import ultralytics
            elif dep == "pickle":
                import pickle
            elif dep == "os":
                import os
            elif dep == "typing":
                import typing
            elif dep == "tensorflow":
                import tensorflow
            elif dep == "sklearn":
                import sklearn
            elif dep == "joblib":
                import joblib
            print(f"✅ {dep} - Available")
        except ImportError as e:
            print(f"❌ {dep} - Missing: {e}")
            missing.append(dep)
    
    if missing:
        print(f"\nMissing dependencies: {', '.join(missing)}")
        print("Please install them using: pip install " + " ".join(missing))
        return False
    else:
        print("\n✅ All dependencies are available!")
        return True

if __name__ == "__main__":
    check_dependencies()