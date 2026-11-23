DOMAIN_TESTS = [
    {
        "id": "TC-DK-01",
        "prompt": "What is SNMP?",
        "expected": "Define SNMP as Simple Network Management Protocol and explain that it is used to monitor/manage network devices."
    },
    {
        "id": "TC-DK-02",
        "prompt": "What is the difference between a MIB and an OID?",
        "expected": "Explain that a MIB defines/organizes managed objects while an OID uniquely identifies a managed object or instance."
    },
    {
        "id": "TC-DK-03",
        "prompt": "What are the five areas of FCAPS?",
        "expected": "Identify Fault, Configuration, Accounting, Performance, and Security."
    },
    {
        "id": "TC-DK-04",
        "prompt": "What does ifAdminStatus = up and ifOperStatus = down indicate?",
        "expected": "Explain that the interface is administratively enabled but is not operational."
    },
    {
        "id": "TC-DK-05",
        "prompt": "Why would an administratively enabled interface still be down?",
        "expected": "Give reasonable causes such as cable/physical link, remote endpoint, transceiver/hardware, speed/duplex, or configuration/data-link issues."
    },
    {
        "id": "TC-DK-06",
        "prompt": "How does SNMP polling work?",
        "expected": "Explain that the manager initiates periodic requests for OIDs and the agent responds; polling is different from unsolicited traps."
    },
    {
        "id": "TC-DK-07",
        "prompt": "Why is my router not working?",
        "expected": "Recognize that the question is ambiguous/incomplete and ask for clarification or additional diagnostic information before identifying one definite cause."
    },
    {
        "id": "TC-DK-08",
        "prompt": "SNMP traps are sent only when the manager polls the device, correct?",
        "expected": "Reject the false premise and explain that traps are unsolicited notifications sent by the SNMP agent."
    }
]
