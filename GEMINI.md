# Gemini Bot Development Log

This file documents the steps taken by the Gemini assistant to develop and modify the bot.

## Initial Bot Implementation

1.  **Requirement Analysis**: Read and analyzed `README.md` to understand the core requirements for the tourist guide bot.

2.  **Database Schema Update**:
    *   Modified `src/db/init_db.py` to create a more comprehensive database schema.
    *   The `users` table was updated to include `first_name`, `last_name`, `active` status, `interests`, and `time`.
    *   A new `locations` table was added to store user location data with columns for `latitude`, `longitude`, and `live_period`.

3.  **Database Methods Update**:
    *   Updated `src/db/users_methods_db.py` to align with the new database schema.
    *   The `add_user` function was modified to accept additional user details.
    *   Added new functions: `update_user_active`, `select_user_active`, `add_location`, and `select_location` to handle location tracking and user status.

4.  **Conversation Flow Implementation**:
    *   Created a new handler file `src/handlers/handlers.py` to manage the main user interaction.
    *   Implemented a state machine using `aiogram.fsm.context.FSMContext` to guide the user through the process of providing their interests, available time, and location.

5.  **Conflict Resolution & Cleanup**:
    *   Identified and resolved a conflict where the `/start` command was defined in both `src/handlers/commands.py` and the new `src/handlers/handlers.py`.
    *   The conflicting handler was removed from `commands.py` to ensure the main conversational flow is triggered correctly.
    *   Deleted the obsolete `handlers-template.py` file to keep the project repository clean.

6.  **UX Improvements**:
    *   Updated the location request handler in `src/handlers/handlers.py`.
    *   The bot now provides a more helpful message if location services are potentially disabled on the user's device.
    *   The location request button is now correctly removed from the keyboard after the user shares their location.

7.  **Processing Animation**:
    *   Added a processing animation to improve user experience during route generation.
    *   A GIF is sent to the user after the `ROUTE_GENERATION_MESSAGE` and is deleted after the route is sent.

## Interest-Based Category Filtering

1.  **RAG System Enhancement**:
    *   Modified `src/ai/rag_logic.py` to improve the relevance of location recommendations.
    *   The system now first identifies the most relevant object category from `categories.txt` based on the user's interests using a call to the Mistral model.
    *   It then filters the dataset to include only objects from that specific category.
    *   A temporary FAISS index is created for the filtered data, and the semantic search is performed on this smaller, more relevant subset.
    *   A fallback mechanism was added to search the entire dataset if no objects are found in the selected category, ensuring the user always receives a response.

2.  **Refined Search Logic**:
    *   Further improved the RAG system's logic in `src/ai/rag_logic.py`.
    *   After filtering the dataset by category, the `category_id` column is now removed.
    *   This ensures that the vector search is performed only on the descriptive data of the objects, leading to more accurate and relevant results.

3.  **Category Selection Logic Refactoring**:
    *   Refactored the category selection logic in `src/ai/rag_logic.py`.
    *   The categories are now stored in a dictionary within the code instead of an external `categories.txt` file.
    *   The AI prompt has been updated to request category IDs instead of names.
    *   The filtering logic now uses these IDs to filter the DataFrame.
    *   The `categories.txt` file has been deleted.

## Code Refactoring and UX Improvements

1.  **Handler Refactoring and Keyboard Simplification**:
    *   Refactored `src/handlers/handlers.py` to reduce code duplication in `process_location` and `process_manual_location` by creating a new `_generate_and_send_route` function.
    *   Simplified the user interface by replacing the two buttons ("Оставить маршрут" and "Переделать маршрут") with a single "Составить новый маршрут" button. When pressed, this button is removed, and the bot restarts the conversation flow, providing a cleaner and more intuitive user experience.

2.  **Improved "New Route"-Button Behavior**:
    *   Modified the `remake_route_callback` function in `src/handlers/handlers.py`.
    *   Instead of deleting the message with the previous route, the "Составить новый маршрут" button is now removed by editing the message and setting the `reply_markup` to `None`.
    *   This change keeps the previous route visible for the user as a reference.

3.  **UI and Message Refactoring**:
    *   Moved all user-facing messages, inline keyboard buttons, and reply keyboard buttons from `src/handlers/handlers.py` to `content/messages.py`, `content/buttons.py`, and `content/keyboards.py` respectively.
    *   This centralizes all UI and text elements, making the application easier to manage and localize.

4.  **Handler Modularization**:
    *   Moved command and callback handlers from `src/handlers/handlers.py` to `src/handlers/commands.py` and `src/handlers/callbacks.py` respectively.
    *   This separation of concerns improves the project structure and makes the codebase more modular and maintainable.

## Localization

1.  **Russian Language Translation**:
    *   Translated all user-facing messages and internal logs to Russian in `src/handlers/handlers.py` and `src/ai/rag_logic.py`.
    *   This ensures a fully localized experience for users and standardizes logs for developers.

## Features

1.  **Interactive Map Generation**:
    *   Implemented a new feature to generate and send an interactive map with the recommended locations.
    *   The `folium` library is used to create a map with markers for each point of interest.
    *   The map is saved as an HTML file and sent to the user as a document, providing a visual representation of the route.

## Bug Fixes

1.  **Category Parsing Error**:
    *   Fixed a bug where the application would crash if the AI returned a comma-separated list of category IDs.
    *   The parsing logic in `src/ai/rag_logic.py` was updated to use a regular expression to correctly handle both comma-separated and newline-separated formats.