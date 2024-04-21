const express = require('express');
const path = require('path');
const bcrypt=require("bcrypt");
const {spawn} = require('child_process')

const app = express();
const collection=require('./config')

//connect data to json format

app.use(express.json());

app.use(express.urlencoded({extended:false}));

// Set the view engine to EJS
app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, '..', 'views'));

// Serve static files from the/Users/lagnikadagur/Desktop/chatbot/chatbot/chatbot_background.jpeg 'public' directory
app.use(express.static('public'));

// Routes
app.get('/', (req, res) => {
    // Render the login page
    res.render('login');
});

app.get('/run-python', (req, res) => {
    const pythonProcess = spawn('python', ['/home/mohit/WebDev Projects/CareerCatalyst/PythonScript/gui.py']);
  
    pythonProcess.stdout.on('data', (data) => {
      console.log(`Python script output: ${data}`);
    });
  
    pythonProcess.stderr.on('data', (data) => {
      console.error(`Error in Python script: ${data}`);
    });
  
    pythonProcess.on('close', (code) => {
      console.log(`Python script exited with code ${code}`);
    });
  });

app.get('/signup', (req, res) => {
    // Render the signup page
    res.render('signup');
});

//Register

app.post("/signup", async (req, res) => {
    const data = {
        name: req.body.username,
        password: req.body.password
    }

    const existingUser = await collection.findOne({ name: data.name });
    if (existingUser) {
        res.send("User already exists. Please choose a different username.");
    } else {
        // Hash the password using bcrypt
        const saltRounds = 10;
        const hashedPassword = await bcrypt.hash(data.password, saltRounds);
        data.password = hashedPassword; // Corrected typo here

        const userdata = await collection.insertMany(data);
        console.log(userdata);
    }
});

// Error handling middleware
app.use((err, req, res, next) => {
    console.error(err.stack);
    res.status(500).send('Something went wrong!');
});

//login user
// login user
app.post("/login", async (req, res) => {
    try {
        const check = await collection.findOne({ name: req.body.username });
        if (!check) {
            res.send("User name not found");
        }

        // compare hashed password
        const isPasswordMatch = await bcrypt.compare(req.body.password, check.password);
        if (isPasswordMatch) {
            res.render("home");
        } else {
            res.send("Wrong password");
        }
    } catch (error) { // Added the catch block
        res.send("Wrong details");
    }
});

// Start the server
const port = process.env.PORT || 5000;
app.listen(port, () => {
    console.log(`Server is running on port ${port}`);
});
