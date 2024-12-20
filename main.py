steps = {
    "Dropped Calls":"""
    1.	Signal Strength Analysis: 
        o	Use network monitoring tools to assess signal levels.
        o	Adjust antenna orientations or upgrade tower hardware where necessary.
    2.	Congestion Management: 
        o	Redistribute traffic loads by optimizing network topology.
        o	Enable dynamic resource allocation protocols.
    3.	Hardware Diagnostics: 
        o	Inspect and replace faulty routers, switches, or base stations.
    """ ,
    "Poor Voice Quality":"""
    1.	Packet Loss Mitigation: 
        o	Monitor packet loss using tools like Wireshark.
        o	Repair damaged cables or optimize routing paths.
    2.	Bandwidth Optimization: 
        o	Prioritize voice traffic via QoS settings.
        o	Expand network capacity during peak hours.
    3.	Codec Reconfiguration: 
        o	Ensure compatible codecs are deployed across the network.
        o	Test and switch to higher-quality codecs when feasible.    
    """,
    "Echo During Calls":"""
    1.	Device Replacement: 
        o	Test and replace faulty headsets or microphones.
    2.	Network Jitter Management: 
        o	Implement jitter buffers on VoIP equipment.
        o	Monitor and stabilize jitter using NOC tools.
    3.	Acoustic Feedback Fixes: 
        o	Adjust audio gain levels on devices and systems.    
    """,
    "Latency and Delay":"""
        1.	Routing Optimization: 
            o	Analyze routing paths and eliminate redundant hops.
            o	Implement MPLS (Multi-Protocol Label Switching) for efficient data flow.
        2.	QoS Configuration: 
            o	Prioritize voice traffic over data traffic.
            o	Implement traffic shaping and bandwidth allocation policies.
        3.	Upgrading Infrastructure: 
            o	Replace legacy systems with modern, high-speed hardware
    """,
    "No Dial Tone":"""
    1.	Verify Physical Connections:
        o	Check that all cables are securely connected to the telephone or VoIP device.
        o	Inspect the wall jack and ensure proper insertion.
    2.	Restart Equipment:
        o	Power cycle the phone, modem, or router to refresh device settings.
    3.	Check Service Provider Status:
        o	Contact the service provider to confirm there are no ongoing outages.
    4.	Test Alternate Lines or Devices:
        o	Use a different phone to rule out device-specific issues.
    5.	Inspect Internal Wiring:
        o	Check for visible damage to wires or jacks. Escalate to technical teams for further inspection if necessary.
    """,
    "Failure to Connect":"""
    1.	Verify Configuration Settings:
        o	Confirm that devices are configured correctly for the network or SIP/VoIP settings.
    2.	Review Network Availability:
        o	Ensure the device is connected to an active network with adequate bandwidth.
    3.	Update Firmware:
        o	Apply the latest firmware or software updates to fix known bugs.
    4.	Check Port Access:
        o	Validate that required ports (e.g., for VoIP: 5060 for SIP) are open and not blocked by firewalls.
    5.	Monitor for Packet Loss or Latency:
        o	Use tools like ping or traceroute to diagnose network instability. Resolve identified issues through routing adjustments or bandwidth prioritization.
    """,
    "Service Outage":"""
    1.	Confirm Outage Scope:
        o	Determine whether the issue affects individual users, specific regions, or the entire network.
    2.	Engage Service Providers:
        o	Report the issue to internet or telecom providers for immediate investigation.
    3.	Switch to Backup Services:
        o	Activate backup internet lines or alternate systems where available.
    4.	Notify Stakeholders:
        o	Inform affected users about the outage and provide an estimated time for resolution.
    5.	Implement Temporary Measures:
        o	Redirect traffic to unaffected systems or re-route calls through alternate servers.
    6.	Post-Outage Diagnostics:
        o	After resolution, analyze logs to identify root causes and prevent recurrence
    """,
    "Blank Response After Calling":"""
    1.	Test Audio Hardware:
        o	Verify functionality of microphones, speakers, or headsets.
    2.	Inspect Codec Compatibility:
        o	Ensure devices are using compatible codecs for voice data transmission.
    3.	Diagnose NAT Traversal Issues:
        o	Configure NAT settings or use tools like STUN/TURN servers to resolve one-way audio issues.
    4.	Monitor Network Traffic:
        o	Check for congestion or dropped packets that may affect call quality.
    5.	Update Session Protocol Configurations:
        o	Review SIP or H.323 settings for errors.
    """,
    "Intermittent Connectivity":"""
    1.	Monitor Network Metrics:
        o	Use monitoring tools to track latency, jitter, and packet loss.
    2.	Identify Peak Usage Periods:
        o	Analyze patterns of connectivity loss to determine correlation with high traffic times.
    3.	Inspect Hardware:
        o	Test and replace routers, switches, or access points exhibiting instability.
    4.	Optimize QoS Settings:
        o	Adjust Quality of Service settings to prioritize voice or critical data traffic.
    5.	Upgrade Infrastructure:
        o	Replace outdated equipment or increase bandwidth to support growing demands
    """,
    "Device Unresponsiveness":"""
    1.	Check Power Supply: 
        o	Verify the device is receiving power and that cables are securely connected.
        o	Replace damaged power adapters or batteries.
    2.	Restart Devices: 
        o	Power cycle the affected hardware to clear temporary errors.
    3.	Inspect Connections: 
        o	Ensure all physical connections (Ethernet, fiber) are secure and functional.
    4.	Verify Device Configuration: 
        o	Review and correct configuration settings to ensure compatibility with the network.
    5.	Test with Alternate Devices: 
        o	Replace the unresponsive device with a backup to isolate hardware-specific issues
    """,
    "Network Downtime":"""
    1.	Scope the Outage: 
        o	Determine whether the issue affects a single device, a segment, or the entire network.
    2.	Check ISP and External Dependencies: 
        o	Verify the status of internet or external service providers.
    3.	Implement Failover Mechanisms: 
        o	Switch to backup lines or redundant systems to restore connectivity.
    4.	Inspect Core Network Equipment: 
        o	Test routers, switches, and firewalls for functionality and configuration accuracy.
    5.	Rebuild Network Routes: 
        o	Clear corrupted routes and reestablish connections to affected nodes.
    """,
    "Degraded Performance":"""
    1.	Monitor Network Traffic: 
        o	Analyze bandwidth usage to detect congestion or bottlenecks.
    2.	Upgrade Bandwidth: 
        o	Increase capacity to accommodate higher traffic demands.
    3.	Optimize Device Settings: 
        o	Configure QoS settings to prioritize critical traffic.
    4.	Inspect and Replace Aging Equipment: 
        o	Identify and replace devices nearing end-of-life or showing performance degradation.
    5.	Apply Firmware Updates: 
        o	Ensure all devices run the latest stable firmware.
    """,
    "Intermittent Failure":"""
    1.	Monitor for Patterns: 
        o	Use logs and monitoring tools to identify recurring triggers for failures.
    2.	Test Environmental Conditions: 
        o	Check for temperature, humidity, or electrical issues affecting hardware.
    3.	Inspect Network Links: 
        o	Test cables, ports, and connectors for physical or signal issues.
    4.	Analyze Jitter and Latency: 
        o	Use diagnostic tools to measure and resolve variations in performance.
    5.	Deploy Redundant Systems: 
        o	Implement redundancy to minimize disruptions during failures.
    """,
    "Hardware Malfunction":"""
    1.	Identify Faulty Components: 
        o	Use diagnostic tools to detect hardware failures (e.g., failing hard drives or memory).
    2.	Replace Faulty Parts: 
        o	Swap out defective components such as NICs, CPUs, or power supplies.
    3.	Inspect Physical Damage: 
        o	Check for signs of wear and tear, corrosion, or overheating.
    4.	Run Hardware Diagnostics: 
        o	Use vendor-provided tools to analyze and resolve hardware issues.
    5.	Test After Replacement: 
        o	Validate functionality with tests after repairs or replacements.
    """,
    "Route Flapping":"""
    1.	Identify Flapping Routes: 
        o	Use monitoring tools to pinpoint frequently changing routes.
    2.	Inspect Network Links: 
        o	Test physical links and cables for stability and connectivity.
    3.	Adjust Route Timers: 
        o	Modify protocol timers (e.g., BGP hold time) to stabilize route announcements.
    4.	Filter Unstable Routes: 
        o	Apply route dampening policies to suppress flapping routes temporarily.
    5.	Replace Faulty Hardware: 
        o	Identify and replace malfunctioning routers or interfaces causing instability.
    """,
    "Misconfigured Routes":"""
    1.	Review Configuration Files: 
        o	Audit router configurations for syntax errors or incorrect entries.
    2.	Verify Address Schemes: 
        o	Ensure IP addresses and subnet masks align with the intended design.
    3.	Test with Traceroute: 
        o	Use traceroute to confirm the actual path matches the desired route.
    4.	Apply Route Filters: 
        o	Implement access control lists (ACLs) or prefix lists to block unintended routes.
    5.	Rollback Changes: 
        o	Revert recent configuration changes if misconfigurations persist
    """,
    "Routing Table Overflow":"""
    1.	Analyze Route Entries: 
        o	Use diagnostic commands to inspect the size and content of the routing table.
    2.	Aggregate Routes: 
        o	Combine multiple routes into summarized entries to reduce table size.
    3.	Implement Route Limits: 
        o	Configure maximum route limits to prevent overflow.
    4.	Offload to Higher-Capacity Devices: 
        o	Replace overloaded routers with models capable of handling larger tables.
    5.	Engage Service Providers: 
        o	Coordinate with ISPs to optimize route advertisements.
    """,
    "Latency or Delays":"""
    1.	Analyze Path Performance: 
        o	Use tools like ping and traceroute to measure delays and identify bottlenecks.
    2.	Optimize Routing Policies: 
        o	Adjust routing metrics to prioritize faster paths.
    3.	Inspect Bandwidth Utilization: 
        o	Ensure sufficient bandwidth for routing protocols and data flow.
    4.	Enable Fast Convergence: 
        o	Configure protocols for rapid failover and path recalculation.
    5.	Upgrade Links: 
        o	Increase capacity of underperforming links.
    """,
    "Routing Protocol Failure":"""
    1.	Check Protocol Status: 
        o	Use diagnostics to verify the operational status of routing protocols (e.g., "show ip bgp summary").
    2.	Synchronize Configurations: 
        o	Ensure protocol settings are consistent across devices.
    3.	Reset Neighbor Relationships: 
        o	Reestablish peering sessions by restarting affected protocols.
    4.	Inspect Authentication Settings: 
        o	Verify authentication credentials for secure protocol exchanges.
    5.	Debug Protocol Events: 
        o	Use debugging commands to trace protocol behavior and resolve issues.
    """,
    "Complete Service Downtime":"""
    1.	Verify Power Supply: 
        o	Check for power issues or failures at the equipment or facility level.
    2.	Inspect Physical Connections: 
        o	Ensure cables and hardware are properly connected and undamaged.
    3.	Reboot Critical Systems: 
        o	Restart affected servers, routers, or switches to refresh operations.
    4.	Engage Service Providers: 
        o	Report outages to ISPs or other service providers for investigation.
    5.	Activate Backup Systems: 
        o	Switch to failover systems to restore temporary functionality.
    """,
    "Partial Outage":"""
    1.	Identify Affected Areas: 
        o	Use monitoring tools to locate specific regions or functionalities impacted.
    2.	Review Configuration Settings: 
        o	Audit system and device configurations for anomalies or recent changes.
    3.	Restart Services: 
        o	Restart individual services or processes exhibiting issues.
    4.	Patch Software: 
        o	Apply updates to resolve known bugs or vulnerabilities causing the problem.
    5.	Isolate Faulty Components: 
        o	Replace or repair devices contributing to the outage.
    """,
    "Intermittent Service":"""
    1.	Monitor Service Patterns: 
        o	Track uptime and downtime intervals to identify trends.
    2.	Inspect Network Stability: 
        o	Check for signal interference, fluctuating bandwidth, or packet loss.
    3.	Test and Replace Hardware: 
        o	Identify malfunctioning devices and replace them if necessary.
    4.	Optimize Network Settings: 
        o	Adjust load balancing and Quality of Service (QoS) configurations.
    5.	Engage Vendors: 
        o	Coordinate with hardware or software providers to diagnose underlying issues.
    """,
    "Slow Performance":"""
    1.	Measure Network Traffic: 
        o	Use tools to identify bottlenecks and congestion points.
    2.	Upgrade Bandwidth: 
        o	Increase capacity to accommodate higher demand.
    3.	Optimize Routing Paths: 
        o	Reconfigure routing to prioritize efficient data flow.
    4.	Enable Caching: 
        o	Implement caching solutions to reduce latency for frequently accessed resources.
    5.	Inspect Background Processes: 
        o	Identify and disable non-critical processes consuming excessive resources.
    """,
    "Unauthorized Disconnections":"""
    1.	Analyze Logs: 
        o	Review system and application logs for errors or unauthorized activity.
    2.	Inspect Firewall and Security Policies: 
        o	Verify that rules are not inadvertently blocking connections.
    3.	Update Access Credentials: 
        o	Change credentials to prevent unauthorized access.
    4.	Run Security Scans: 
        o	Check for malware or suspicious activities disrupting services.
    5.	Reset Network Devices: 
        o	Clear device configurations causing unexpected disconnections.
    """,
    "unauthorized access":"""
    1.	Isolate Affected Systems: 
        o	Disconnect compromised devices from the network to contain the breach.
    2.	Reset Credentials: 
        o	Force password resets for affected user accounts.
    3.	Audit Logs: 
        o	Review access logs to identify unauthorized activities and entry points.
    4.	Patch Vulnerabilities: 
        o	Apply updates or patches to close exploited security gaps.
    5.	Enable Multi-Factor Authentication (MFA): 
        o	Add MFA to enhance access security.
    """,
    "malware infections":"""
    1.	Quarantine Infected Devices: 
        o	Isolate devices to prevent the spread of malware.
    2.	Run Anti-Malware Tools: 
        o	Perform deep scans using trusted anti-malware solutions.
    3.	Analyze Malware Behavior: 
        o	Investigate malware’s actions to understand its impact.
    4.	Restore from Backups: 
        o	Revert affected systems to their last secure state using backups.
    5.	Educate Users: 
        o	Train users on identifying and avoiding malicious links and attachments.
    """,
    "Phishing Attacks":"""
    1.	Alert Affected Users: 
        o	Notify users targeted by phishing campaigns.
    2.	Block Malicious URLs: 
        o	Use security tools to block identified phishing sites.
    3.	Monitor for Compromised Accounts: 
        o	Check for signs of account misuse or unauthorized activities.
    4.	Report to Authorities: 
        o	Inform cybersecurity authorities or platforms hosting the phishing domains.
    5.	Enhance Email Filtering: 
        o	Improve spam filters to reduce phishing attempts reaching users.
    """,
    "denial of services":"""
    1.	Identify Attack Traffic: 
        o	Use monitoring tools to detect anomalous traffic patterns.
    2.	Apply Traffic Filtering: 
        o	Configure firewalls and intrusion prevention systems to block malicious IPs.
    3.	Enable Rate Limiting: 
        o	Set limits on traffic to prevent resource overload.
    4.	Engage DDoS Mitigation Services: 
        o	Work with third-party services specialized in DDoS mitigation.
    5.	Review Incident Post-Mortem: 
        o	Analyze logs to identify attack vectors and strengthen defenses.
    """,
    "data breaches":"""
    1.	Contain Breach Scope: 
        o	Restrict access to affected systems and data repositories.
    2.	Engage Forensic Experts: 
        o	Work with cybersecurity experts to investigate the breach.
    3.	Notify Stakeholders: 
        o	Inform affected parties and regulatory bodies as required.
    4.	Enhance Data Security: 
        o	Encrypt sensitive data and restrict access permissions.
    5.	Implement Monitoring Tools: 
        o	Deploy real-time monitoring to detect and respond to future breaches.
    """,
    "common":"""
    1.Verify Configuration and Connectivity: Ensure devices are correctly configured and connected to an active network.
    2.Check Network and Device Health: Assess network availability and check for firmware updates or port access issues.
    3.Engage Service Providers/Support: Contact the necessary service providers or technical support for immediate troubleshooting.
    4.Implement Temporary Solutions: Use backup services or reroute traffic to maintain operations while resolving the issue.
    5.Monitor and Diagnose Post-Incident: Review logs and diagnostic tools to identify the root cause and prevent future occurrences.
    """
}

# Create a new dictionary with lowercase keys
steps_lower = {key.lower(): value for key, value in steps.items()}





from fpdf import FPDF
from PyPDF2 import PdfMerger

import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings   ######## for embedding (instead of ollama embedding)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import pandas as pd
import numpy as np
import joblib
import time
from dotenv import load_dotenv
load_dotenv()

# Load the trained model
model = joblib.load('model.pkl')

# Read the dataset from a local file
df = pd.read_csv('dataset.csv')


# Streamlit page title
st.title("Incident Root Cause Analysis")

# Text inputs for user inputs
incident_id_input = st.text_input("Incident ID", "") 
actions_taken = st.text_area("Actions Taken / Current Progress", "")
issues_faced = st.text_area("Issues Faced During Implementation", "")
c=0
# Button to generate output
if st.button("Generate PDF"):
    # st.write("## Analysis Output")

    # Ensure the user provides an Incident ID
    if incident_id_input.strip() == "":
        st.error("Please provide a valid Incident ID.")
    else:
        # Filter the dataset based on the input Incident ID
        # Ensure both the input and dataset values are stripped of leading/trailing spaces and properly compared
        filtered_df = df[df['Incident ID'].astype(str).str.strip() == incident_id_input.strip()]


        if filtered_df.empty:
            st.error("No matching Incident ID found in the dataset.")
        else:
            # Process only the matching row
            with open("analysis_output.txt", "w") as file:
                for _, row in filtered_df.iterrows():
                    # Access columns: 'Incident ID', 'Incident Type', and other necessary columns
                    incident_id = row['Incident ID']
                    incident_type = row['Incident Type']
                    jitter = row['Jitter']
                    jitter = int(jitter.replace("ms", ""))
                    latency = row['Latency']
                    latency = int(latency.replace("ms", ""))
                    packet_loss = row['Packet Loss']
                    packet_loss = int(packet_loss.replace("%", ""))
                    signal_strength = row['Signal Strength']

                    # Determine steps based on incident type
                    if incident_type.lower() in steps_lower:
                        step = steps_lower[incident_type.lower()]
                    else:
                        step = steps_lower["common"]

                    # Prepare input features for prediction
                    input_features = np.array([[jitter, latency, packet_loss, signal_strength]])

                    # Convert input features to DataFrame with appropriate column names
                    feature_names = ['Jitter', 'Latency', 'Packet Loss', 'Signal Strength']
                    input_features_df = pd.DataFrame(input_features, columns=feature_names)

                    # Make the prediction
                    predicted_issue = model.predict(input_features_df)

                    # Display the details for the incident
                    # st.write(f"### Incident ID: {incident_id}")
                    # st.write(f"- Incident Type: {incident_type}")
                    # st.write(f"- Root Cause: {predicted_issue[0]}")
                    # st.write(f"- Steps: {step}")

                    # # Print all columns dynamically
                    # st.write("#### Incident Details")
                    # for column_name, value in row.items():
                    #     st.write(f"- {column_name}: {value}")

                    # st.write("---")

                    # Append output to the text file
                    file.write(f"Incident ID: {incident_id}\n")
                    file.write(f"Incident Type: {incident_type}\n")
                    file.write(f"Root Cause: {predicted_issue[0]}\n")
                    file.write(f"Steps: {step}\n")
                    file.write("Incident Details:\n")
                    for column_name, value in row.items():
                        file.write(f"- {column_name}: {value}\n")

                # Append user inputs to the text file
                file.write(f"Actions Taken / Current Progress: {actions_taken}\n")
                file.write(f"Issues Faced During Implementation: {issues_faced}\n")
                c=1

#start of srihari's model



if(c==1):
        
    # Load environment variables
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
    groq_api_key = os.getenv("GROQ_API_KEY")

    # LLM and Prompt setup
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")
    
    # You are an experienced Network Operations Center (NOC) engineer. Generate a detailed incident report based on the provided input. Include the following:

    # A clear and concise summary of the issue.
    # Detailed steps taken to diagnose and resolve the issue.
    # Key metrics or data relevant to the incident.
    # The root cause of the problem.
    # The resolution and any follow-up actions required.
    # Additional recommendations to prevent recurrence.
    # Ensure the documentation is professional, thorough, and suitable for technical review.
    prompt = ChatPromptTemplate.from_template(
        """

    follow the following template for generating the incident report

    Incident Report
    Incident Identification
    - ID:
    - Timestamp:
    - Incident Type:
    - Call Type:
    - Source Number:
    - Destination Number:
    - Affected Systems:

    Incident Description
    2-3 lines about the incident

    Technical Details
    - Device Type: 
    - Network Segment: 
    - Codec Used:

    Incident Report
    Impact Assessment
    - Severity Level: 
    - Users Impacted: 
    - Impact Area: 
    - Business Impact:

    Root Cause Analysis
        give a brief description of the root cause analysis

    Actions Taken
    Incident Report
        Report for the oncident

    Conclusion
        give a 3 line conclusion

    <context>
    {context}
    <context>
    Question:{input}
    """
    )

    # Function to create vector embeddings
    def create_vector_embeddings():
        if "vectors" not in st.session_state:
            st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            st.session_state.loader = PyPDFDirectoryLoader("dataset")   # Data ingestion
            st.session_state.docs = st.session_state.loader.load()      # Document loading
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

    # Load user input from a file

    create_vector_embeddings()
    with open('analysis_output.txt', 'r') as file:
        file_content = ''.join(file.readlines())

    user_prompt = file_content

    # Initialize session_state.vectors to prevent attribute errors
    if "vectors" not in st.session_state:
        st.session_state.vectors = None

        # Ensure vectors are initialized before accessing
    if st.session_state.vectors:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()  # Safe access
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        start = time.process_time()
        response = retrieval_chain.invoke({'input': user_prompt})
        print(f"Response time is: {time.process_time() - start}")
        # st.write(response['answer'])
        ### Streamlit  
       

        class PDF(FPDF):
            def __init__(self):
                super().__init__()
                self.set_auto_page_break(auto=True, margin=15)

            def add_content(self, text):
                self.add_page()
                self.set_font("Arial", size=12)
                self.multi_cell(0, 10, text)

        def append_to_pdf(existing_pdf, new_content):
            # Create a temporary PDF with the new content
            temp_pdf = "temp_append.pdf"
            pdf = PDF()
            pdf.add_content(new_content)
            pdf.output(temp_pdf)
                # Merge the original and the new PDF
            merger = PdfMerger()
            merger.append(existing_pdf)
            merger.append(temp_pdf)            
            merger.write(existing_pdf)
            merger.close()
            print(f"Appended content saved to: {existing_pdf}")

            # Append the new response to the main PDF
        append_to_pdf("initial.pdf", response['answer'])
        st.write("PDF Document is Ready")
