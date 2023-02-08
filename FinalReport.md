# Machine Learning Algorithms on an Embedded Processor

## Table of Contents
- [Abstract:](#abstract-)
- [Introduction:](#introduction-)
- [Object Detection](#object-detection)
- [Text Detection](#text-detection)
- [Repository Organization and Management](#repository-organization-and-management)
- [Domain Research](#domain-research)
- [Domain Knowledge](#domain-knowledge)
- [Areas of Technical Growth](#areas-of-technical-growth--)
- [Software Engineering Code of Ethics and Professional Practice](#software-engineering-code-of-ethics-and-professional-practice)
- [Teamwork Reflection](#teamwork-reflection)
- [Conclusion](#conclusion)

## Abstract:

In today’s world, it can be hard to discern the role machine learning plays in your everyday life. From virtual chatbot helpers on your phone to the predictive algorithms in your FaceBook feed, machine learning has become an industry solution standard in computer science. Areas such as speech, text, and image recognition, symptom pattern recognition in healthcare, and predictive analytics in sports are all made more robust through the application of machine learning. It may be easy to assume that machine learning must use a vast amount of resources and hardware to achieve the power and functionality that it provides, but this would be folly. For our capstone, we explored the industry standards for running object detection and text detection machine learning algorithms on embedded ARM processors.

The project itself is an exploration into the industry standards available currently for creating object and text detection software on embedded systems, specifically Pilinix/PetaLinux. The goal of this application is to have functionality for text recognition and object detection via a VGA data stream (a picture, or video). Additionally, the application is to have the functionality to verify if a segment of the VGA data stream is displaying the expected data or not, and to train the algorithms to identify new objects. To ensure the program is compatible on an embedded system, the program is written in Python, and the machine learning algorithms from TensorFlow, both of which run on embedded hardware.

## Introduction:

The machine learning (ML) algorithms and their functionality are the primary focus of this program. The application accepts a video graphics array (VGA) as its input. Depending on the ML model picked, the application then performs an object or text detection on the VGA input and outputs the detections to a new image or video file. The application processes the VGA input using either Cv2 or Pillow, and passes the VGA to the ML model for processing. Using a label file which contains a list of the trained possible detections, the ML model creates sets of metadata for each detection found from the label file. Matplotlib, using the metadata from the ML model, plots where each detection took place within the VGA input, and draws the detection with the label and confidence score. A new image or video file is created with the detections, and is then displayed to the user in the application UI.

## Object Detection

TensorFlow Hub is a repository of trained machine learning models that can be accessed and used in different environments. TensorFlow Hub contains an object detection model which we used for this project. This model came with a small label map file that we expanded to hold around one hundred objects. We fed the user specified input into the TensorFlow Hub model which gave us our predictions for what was being detected. Then we used Matplotlib to draw detection boxes around those detections and added a label and a confidence to each. The Gui will output the top three detections with the highest confidences. To process the image we used python’s CV2, this helped us change the size and format the image to fit the Gui better. Formatting the output was done with Pillow and the user can browse their system for images that will be used as the VGA input data. 

Figure 1 displays the view of the application after an object detection is performed.



Figure 1: The view of the application displaying the VGA input after an object detection has been performed. Shown in the detections is the label of each detection and the confidence scores to each.

## Text Detection
Optical character recognition (OCR) is a process of recognizing the characters from images using computer vision and machine learning algorithms. Tensorflow’s OCR accomplishes text recognition in two main stages. The first stage takes an image and, using a text detection model, detects the bounding boxes around possible sets of text. Second, Tensorflow’s OCR feeds the processed bounding boxes into a text recognition model to determine the specific characters within the bounding boxes. Tensorflow’s OCR is in a binary number format called half-precision floating-point format, which is a common format used in ML models dealing in image processing. This allows for faster image processing where higher precision is not essential.

Due to the fact that rectangle detection boxes may overlap with additional detections and text, we elected to output the detection data to the terminal. A terminal containing a text detection’s output data is shown in Figure 2.



Figure 2: The view of the application after a text detection is performed. The view shows the VGA input chosen for the detection, and the terminal window with the output data.

## Repository Organization and Management

The organization of our GitHub repository is an area that we could have improved in this project. Our plan before the start of the project was to make a distinct branch for each sprint and issue. However, our work often overlapped and the features we worked on often spanned more than one sprint, which led to us changing the repository organization. In the end, we made a branch for each feature. Another result of this is that our burndown charts were not very readable, due to the issues in ZenHub being so large. If we could do it again, we would try to break each task down into smaller issues to be completed within one sprint more realistically. This would also make using ZenHub more manageable and make our burndown charts more readable.

## Domain Research

There were multiple research ideas that were identified for our project. The main ones were getting a working machine learning model or having to create our own, what machine learning models and modules would work on embedded processors, and understanding how to accept and manipulate VGA input.

We first had to understand what the nomenclature around machine learning, such as the difference between a supervised machine learning algorithm, semi-supervised machine learning algorithm, and unsupervised machine algorithm are. We then had to decide if there were any industry standard solutions available for ML on embedded systems. We initially identified TensorFlow Lite as a solution to go with, but ran into integration issues and abandoned it in favor of regular TF. It was found that both TensorFlow and Python were friendly on embedded systems and had object and text detection models ready at our disposal.

## Domain Knowledge

We initially had trouble creating a machine learning model because it would have taken the majority of our time just to train it and we would not have known if it would work. We decided to look for pretrained models that would work on embedded systems. After many failed models we found the right one for both object and text detection.

Accepting VGA input was an easy yet difficult thing to accomplish. Single images were easy to implement into our models, but videos would have taken an exponentially longer time to process. Single images could take up to ten seconds to complete detections, on a thirty frame per second video that would be three hundred seconds to load detections onto one second of the video along with all of the sections and the output data.

## Areas of Technical Growth

**Brennan:**
During my internship experience, I worked lightly in the field of ML, utilizing TensorFlow and API’s such as FaceAPI to accomplish face tracking. This project expanded my initial knowledge and understanding of the field of neural networks through the process of implementing them in this project. I also gained experience in the field of management, working to lead the team on sprint goals and project deadlines. 

**Austin:**
Coming into this project I wanted to learn more about machine learning and advanced neural networks, being that that is the entire premise of the project. I learned quite a bit about it. Figuring out what models would work depending on how they were trained helped me understand how they work, we had gone through a handful of models that ended up not working and they all were different and taught me something else about how they are trained. My internship was mainly about automating schematics but I had to build a Gui from scratch which helped when it came to the Gui development in our project. Although I had very little experience with what this project encompasses, I believe that this project has given me great knowledge about how machine learning algorithms and advanced neural networks are trained and used in the real world.

**Sam:**
Starting the project, I had never worked with Machine Learning or AI before. Most of the technical work I have done, especially at my internship, was back end work and front end web development. This project gave me the opportunity to work hands on with a Neural Network and expand my computer science knowledge base. I am thankful for this project because I got to work with a technology I otherwise might not have: Machine Learning. I also worked with new tools to create the GUI, such as the grid system in Tkinter.

## Software Engineering Code of Ethics and Professional Practice

**Principle 2: Client and Employer**

2.01. Provide service in their areas of competence, being honest and forthright about any limitations of their experience and education. 

Throughout the project we had been honest and forthright with our employer when running into any limitations and problems. Also all shortcomings during the project were explained to our employer.

2.06. Identify, document, collect evidence and report to the client or the employer promptly if, in their opinion, a project is likely to fail, to prove too expensive, to violate intellectual property law, or otherwise to be problematic. 

Although the project did not fail there were a couple goals that were not reached and overall the project could have been developed in a more efficient manner and this was brought up and presented to our client.

**Principle 3: Product**

3.02. Ensure proper and achievable goals and objectives for any project on which they work or propose. 

All goals were achievable if given more time with more experience, we ran into too many problems that brought the overall quality of our project down and we didn’t achieve a couple goals because of this.

3.10. Ensure adequate testing, debugging, and review of software and related documents on which they work. 

All code was vetted to run on embedded processors and when developed we had a testing process to make sure everything was working properly

**Principle 5: Management**

5.09. Ensure that there is a fair agreement concerning ownership of any software, processes, research, writing, or other intellectual property to which a software engineer has contributed.

The project is property of Array of Engineers and they plan to continue work on it.

**Principle 6: Profession**

6.08. Take responsibility for detecting, correcting, and reporting errors in software and associated documents on which they work. 

All errors were handled and corrected that were found. Any proceeding errors can be taken care of by adding a case for that test.

**Principle 7: Colleagues**

7.02. Assist colleagues in professional development.

We helped each other when necessary and sought help from our client and professor when needed.

**Principle 8: Self**

8.02. Improve their ability to create safe, reliable, and useful quality software at reasonable cost and within a reasonable time.

We all learned at least one thing about how to develop better as a professional and how to efficiently code software with reasonable cost and within a reasonable time.

## Teamwork Reflection

We had weekly meetings with our sponsor to update them with progress on the project. Although we had frequent meetings with the client, we still had some teamwork issues to work through. Communication was a challenge for us throughout the semester. There were times when not everyone was on the same page, or meetings were missed because of poor communication. Our communication improved throughout our time with the project, but it was a valuable learning experience. Our solutions for this problem included having more frequent meetings with just the three team members to catch up on progress and plan ahead and using the group chat more frequently to update each other on our progress.

## Conclusion

Going into the semester, our sponsor viewed this project as a proof of concept, as they were unsure if the specifications of the project were possible. With this in mind, we created many goals, and accepted going in that we most likely would not be able to complete all of them over the course of the semester. However, we were able to complete a large part of what our sponsor asked from us. We created a program to match up with our sponsor’s specifications as closely as possible. Although there are still areas for development on the project, we are proud of the product we delivered. The next steps for another group who might pick this project up in the future include hyperthreading to increase efficiency, as well as giving the user more options for modifying the file they want to do a detection on. The main two ways to do this would be to let the user select a smaller portion of the photo or video on which they want to detect, and letting the user select timestamps within which the program will detect. We did not finish every feature that the final version might include, but we are very satisfied with the product that we delivered.
