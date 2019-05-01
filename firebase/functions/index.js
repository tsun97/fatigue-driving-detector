const functions = require('firebase-functions');
const sgMail = require('@sendgrid/mail');
sgMail.setApiKey('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');

// // Create and Deploy Your First Cloud Functions
// // https://firebase.google.com/docs/functions/write-firebase-functions
//
// exports.helloWorld = functions.https.onRequest((request, response) => {
//  response.send("Hello from Firebase!");
// });

const APP_NAME = 'Fatigue Driving Prevention';

exports.onAddRecord = functions.database.ref('/records/{pushId}').onCreate((snapshot, context) => {
    // Grab the text parameter.
    const newData = snapshot.val();    
    const device = newData.device;
    const time = newData.time;
    const email = newData.email;

    // Send email notification
    const message = {
        from: `detector@fatigue-driving.com`,
        to: `${email}`,
        subject: `Fatigue Driving Warning`,
        text: `Warning: The driver on ${device} fell asleep during driving!\nTime: ${time}`
    };
    sgMail.send(message);
    
    return null;
});
