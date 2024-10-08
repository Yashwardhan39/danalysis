-- ====================================================
-- Schema Creation for JIVI Backend
-- ====================================================

-- Drop tables if they already exist to avoid conflicts
DROP TABLE IF EXISTS UserRemedies CASCADE;
DROP TABLE IF EXISTS Remedies CASCADE;
DROP TABLE IF EXISTS UserMedications CASCADE;
DROP TABLE IF EXISTS Medications CASCADE;
DROP TABLE IF EXISTS UserDoctorRecommendations CASCADE;
DROP TABLE IF EXISTS Doctors CASCADE;
DROP TABLE IF EXISTS UserDiagnoses CASCADE;
DROP TABLE IF EXISTS Diagnoses CASCADE;
DROP TABLE IF EXISTS UserSymptoms CASCADE;
DROP TABLE IF EXISTS Symptoms CASCADE;
DROP TABLE IF EXISTS HealthcareCharts CASCADE;
DROP TABLE IF EXISTS DietCharts CASCADE;
DROP TABLE IF EXISTS Users CASCADE;

-- ====================================================
-- Users Table
-- ====================================================
CREATE TABLE Users (
    UserID SERIAL PRIMARY KEY,
    Name VARCHAR(100) NOT NULL,
    Age INT,
    Gender VARCHAR(10),
    ContactInformation VARCHAR(255),
    RegistrationDate DATE
);

-- ====================================================
-- Symptoms Table
-- ====================================================
CREATE TABLE Symptoms (
    SymptomID SERIAL PRIMARY KEY,
    SymptomDescription VARCHAR(255) NOT NULL
);

-- ====================================================
-- UserSymptoms Table
-- ====================================================
CREATE TABLE UserSymptoms (
    UserSymptomID SERIAL PRIMARY KEY,
    UserID INT REFERENCES Users(UserID) ON DELETE CASCADE,
    SymptomID INT REFERENCES Symptoms(SymptomID) ON DELETE CASCADE,
    DateReported DATE,
    Severity INT, -- Scale from 1 (mild) to 10 (severe)
    Duration VARCHAR(50) -- e.g., '2 days', '1 week'
);

-- ====================================================
-- Diagnoses Table
-- ====================================================
CREATE TABLE Diagnoses (
    DiagnosisID SERIAL PRIMARY KEY,
    DiagnosisName VARCHAR(100) NOT NULL,
    Description TEXT
);

-- ====================================================
-- UserDiagnoses Table
-- ====================================================
CREATE TABLE UserDiagnoses (
    UserDiagnosisID SERIAL PRIMARY KEY,
    UserID INT REFERENCES Users(UserID) ON DELETE CASCADE,
    DiagnosisID INT REFERENCES Diagnoses(DiagnosisID) ON DELETE CASCADE,
    DateDiagnosed DATE,
    ConfidenceScore DECIMAL(5,2) -- Percentage confidence in diagnosis
);

-- ====================================================
-- Doctors Table
-- ====================================================
CREATE TABLE Doctors (
    DoctorID SERIAL PRIMARY KEY,
    Name VARCHAR(100) NOT NULL,
    Specialization VARCHAR(100),
    AvailabilityStatus VARCHAR(50), -- e.g., 'Available', 'Busy'
    ContactInformation VARCHAR(255)
);

-- ====================================================
-- UserDoctorRecommendations Table
-- ====================================================
CREATE TABLE UserDoctorRecommendations (
    RecommendationID SERIAL PRIMARY KEY,
    UserID INT REFERENCES Users(UserID) ON DELETE CASCADE,
    DoctorID INT REFERENCES Doctors(DoctorID) ON DELETE CASCADE,
    RecommendationDate DATE
);

-- ====================================================
-- Medications Table
-- ====================================================
CREATE TABLE Medications (
    MedicationID SERIAL PRIMARY KEY,
    MedicationName VARCHAR(100) NOT NULL,
    Description TEXT,
    DosageInformation VARCHAR(100)
);

-- ====================================================
-- UserMedications Table
-- ====================================================
CREATE TABLE UserMedications (
    UserMedicationID SERIAL PRIMARY KEY,
    UserID INT REFERENCES Users(UserID) ON DELETE CASCADE,
    MedicationID INT REFERENCES Medications(MedicationID) ON DELETE CASCADE,
    StartDate DATE,
    EndDate DATE,
    Dosage VARCHAR(50),
    Frequency VARCHAR(50)
);

-- ====================================================
-- Remedies Table
-- ====================================================
CREATE TABLE Remedies (
    RemedyID SERIAL PRIMARY KEY,
    RemedyName VARCHAR(100) NOT NULL,
    Description TEXT
);

-- ====================================================
-- UserRemedies Table
-- ====================================================
CREATE TABLE UserRemedies (
    UserRemedyID SERIAL PRIMARY KEY,
    UserID INT REFERENCES Users(UserID) ON DELETE CASCADE,
    RemedyID INT REFERENCES Remedies(RemedyID) ON DELETE CASCADE,
    StartDate DATE,
    EndDate DATE
);

-- ====================================================
-- DietCharts Table
-- ====================================================
CREATE TABLE DietCharts (
    DietChartID SERIAL PRIMARY KEY,
    UserID INT REFERENCES Users(UserID) ON DELETE CASCADE,
    CreationDate DATE,
    DietaryRecommendations JSONB
);

-- ====================================================
-- HealthcareCharts Table
-- ====================================================
CREATE TABLE HealthcareCharts (
    HealthcareChartID SERIAL PRIMARY KEY,
    UserID INT REFERENCES Users(UserID) ON DELETE CASCADE,
    Date DATE,
    HealthMetrics JSONB
);

-- ====================================================
-- Inserting Sample Data into Tables
-- ====================================================

-- ----------------------------
-- Users
-- ----------------------------
INSERT INTO Users (Name, Age, Gender, ContactInformation, RegistrationDate) VALUES
('Alice Johnson', 30, 'Female', 'alice.johnson@example.com', '2023-01-15'),
('Bob Smith', 45, 'Male', 'bob.smith@example.com', '2023-03-22'),
('Charlie Davis', 25, 'Non-binary', 'charlie.davis@example.com', '2023-05-10');

-- ----------------------------
-- Symptoms
-- ----------------------------
INSERT INTO Symptoms (SymptomDescription) VALUES
('Fever'),
('Cough'),
('Headache'),
('Fatigue'),
('Sore Throat'),
('Shortness of Breath'),
('Nausea'),
('Dizziness');

-- ----------------------------
-- UserSymptoms
-- ----------------------------
INSERT INTO UserSymptoms (UserID, SymptomID, DateReported, Severity, Duration) VALUES
(1, 1, '2024-09-20', 7, '3 days'),
(1, 3, '2024-09-20', 5, '2 days'),
(2, 2, '2024-09-18', 6, '5 days'),
(2, 4, '2024-09-18', 4, '1 week'),
(3, 5, '2024-09-25', 3, '1 day'),
(3, 7, '2024-09-25', 8, '4 days');

-- ----------------------------
-- Diagnoses
-- ----------------------------
INSERT INTO Diagnoses (DiagnosisName, Description) VALUES
('Common Cold', 'A viral infectious disease of the upper respiratory tract.'),
('Influenza', 'A viral infection that attacks your respiratory system.'),
('Migraine', 'A headache of varying intensity, often accompanied by nausea and sensitivity to light.'),
('Hypertension', 'A condition in which the blood pressure in the arteries is persistently elevated.');

-- ----------------------------
-- UserDiagnoses
-- ----------------------------
INSERT INTO UserDiagnoses (UserID, DiagnosisID, DateDiagnosed, ConfidenceScore) VALUES
(1, 1, '2024-09-21', 85.50),
(2, 2, '2024-09-19', 90.00),
(3, 3, '2024-09-26', 75.25);

-- ----------------------------
-- Doctors
-- ----------------------------
INSERT INTO Doctors (Name, Specialization, AvailabilityStatus, ContactInformation) VALUES
('Dr. Emily Clark', 'General Practitioner', 'Available', 'emily.clark@jivihealth.com'),
('Dr. Michael Brown', 'Pulmonologist', 'Busy', 'michael.brown@jivihealth.com'),
('Dr. Sarah Lee', 'Neurologist', 'Available', 'sarah.lee@jivihealth.com'),
('Dr. David Wilson', 'Cardiologist', 'Available', 'david.wilson@jivihealth.com');

-- ----------------------------
-- UserDoctorRecommendations
-- ----------------------------
INSERT INTO UserDoctorRecommendations (UserID, DoctorID, RecommendationDate) VALUES
(1, 1, '2024-09-21'),
(2, 2, '2024-09-19'),
(3, 3, '2024-09-26');

-- ----------------------------
-- Medications
-- ----------------------------
INSERT INTO Medications (MedicationName, Description, DosageInformation) VALUES
('Paracetamol', 'Pain reliever and a fever reducer.', '500mg every 6 hours'),
('Ibuprofen', 'Nonsteroidal anti-inflammatory drug.', '200mg every 8 hours'),
('Amoxicillin', 'Antibiotic used to treat bacterial infections.', '500mg every 12 hours'),
('Lisinopril', 'Used to treat high blood pressure.', '10mg once daily');

-- ----------------------------
-- UserMedications
-- ----------------------------
INSERT INTO UserMedications (UserID, MedicationID, StartDate, EndDate, Dosage, Frequency) VALUES
(1, 1, '2024-09-21', '2024-09-24', '500mg', 'Every 6 hours'),
(2, 2, '2024-09-19', '2024-09-24', '200mg', 'Every 8 hours'),
(3, 3, '2024-09-26', '2024-10-03', '500mg', 'Every 12 hours'),
(2, 4, '2024-09-19', '2024-10-19', '10mg', 'Once daily');

-- ----------------------------
-- Remedies
-- ----------------------------
INSERT INTO Remedies (RemedyName, Description) VALUES
('Honey Lemon Tea', 'Helps soothe a sore throat and reduce coughing.'),
('Ginger Compress', 'Reduces headache and nausea symptoms.'),
('Steam Inhalation', 'Alleviates nasal congestion and improves breathing.');

-- ----------------------------
-- UserRemedies
-- ----------------------------
INSERT INTO UserRemedies (UserID, RemedyID, StartDate, EndDate) VALUES
(1, 1, '2024-09-21', '2024-09-24'),
(2, 2, '2024-09-19', '2024-09-25'),
(3, 3, '2024-09-26', '2024-09-30');

-- ----------------------------
-- DietCharts
-- ----------------------------
INSERT INTO DietCharts (UserID, CreationDate, DietaryRecommendations) VALUES
(1, '2024-09-21', '{
    "Breakfast": "Oatmeal with fruits",
    "Lunch": "Grilled chicken salad",
    "Dinner": "Steamed vegetables and brown rice",
    "Snacks": "Nuts and yogurt"
}'),
(2, '2024-09-19', '{
    "Breakfast": "Whole grain toast with avocado",
    "Lunch": "Quinoa and black bean bowl",
    "Dinner": "Baked salmon with asparagus",
    "Snacks": "Fruit smoothies"
}'),
(3, '2024-09-26', '{
    "Breakfast": "Scrambled eggs and spinach",
    "Lunch": "Turkey sandwich on whole wheat",
    "Dinner": "Vegetable stir-fry with tofu",
    "Snacks": "Carrot sticks and hummus"
}');

-- ----------------------------
-- HealthcareCharts
-- ----------------------------
INSERT INTO HealthcareCharts (UserID, Date, HealthMetrics) VALUES
(1, '2024-09-21', '{
    "Temperature": "38.5°C",
    "BloodPressure": "120/80",
    "HeartRate": "75 bpm",
    "OxygenSaturation": "98%"
}'),
(1, '2024-09-22', '{
    "Temperature": "37.8°C",
    "BloodPressure": "118/79",
    "HeartRate": "72 bpm",
    "OxygenSaturation": "99%"
}'),
(2, '2024-09-19', '{
    "Temperature": "39.0°C",
    "BloodPressure": "130/85",
    "HeartRate": "80 bpm",
    "OxygenSaturation": "95%"
}'),
(3, '2024-09-26', '{
    "Temperature": "36.7°C",
    "BloodPressure": "115/75",
    "HeartRate": "70 bpm",
    "OxygenSaturation": "97%"
}');
