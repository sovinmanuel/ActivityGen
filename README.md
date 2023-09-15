# Abstract of Thesis

Robotic Process Automation (RPA) enables the creation of bots that automate time-consuming tasks in applications that exhibit repetitive patterns. Process Mining (PM) techniques help identify such patterns and develop better bots by analyzing interaction logs that capture user interaction with the mouse and contain screenshots for each interaction, which captures the state of the desktop. However, the interaction logs can not be directly used to apply PM techniques, and they have to be transformed into a suitable format, namely event logs. Translucent event logs have been proposed as a new format, an extension of event logs containing information about enabled activities extracted from the screenshots. Dedicated PM techniques for translucent event logs outperform traditional ones on event logs, thereby helping improve RPA initiatives. However, current techniques for extracting enabled activities rely on template matching, which leads to unreliable detection. Moreover, no approach exists for creating translucent event logs based on interaction logs automatically, and a manual creation would be time-consuming, as every enabled activity depicted on the screenshots has to be manually extracted. Therefore, we propose ActivityGen, a modular framework with optional user guidance for automatically extracting enabled activities from screenshots, allowing translucent event logs to be created. First, components in a screenshot are detected and classified using Computer Vision (CV) techniques and Machine Learning (ML). After that, these components are filtered on relevancy and associated with an activity name using a generated expressive label conveying the user interaction. We evaluate ActivityGen extensively in quantitative and qualitative aspects regarding detecting components and generating labels for the activities. We present a case study using a real-life workflow that showcases the promising results of ActivityGen.

# Initialization

> Install packages using [*pipenv*](https://pipenv.pypa.io/en/latest/)

> Download the  [*model weights*](https://drive.google.com/drive/folders/1cV4uA4EWwmgCts6oKjR-X7affKEMR6_6?usp=sharing) and put them under the folder '/models'

# Project Structure

    .
    ├── ...
    ├── activitygen                     # Core package folder
    │   ├── activity_name_generator     # Modules for generating & saving activity names, context
    │   ├── compo_classifier            # Modules for classifying detected components and ViSM
    │   ├── compo_detector              # Module for non-text detection and user-guided
    │   ├── merge                       # Module for merging all detected components
    │   └── text_detector               # Module for text detection
    ├── configs                         # Folder for config files to change parameters
    ├── models                          # Folder for model weights 
    └── ...

# Usage

> To use ActivityGen, the the *run_\*.py* files can be modified and executed.
They are the main entry points and show examples how the different modules can be combined.

> *run_batch.py* can be used to run ActivityGen on a folder containing different screenshots.

> The Jupyter notebooks can be executed to run example of the transformation from an interaction log into a translucent event log. <br>
> *process_workflow_open.ipynb* executes open transformation for an example workflow. <br>
> *process_workflow_closed.ipynb* executes closed transformation for an example workflow.


