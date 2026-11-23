"""Checks that the automatic validation gives the right output class on known answers.

Run: python -m pytest -q tests
Each test feeds a hand-written answer (good, partial, wrong, fabricated) to the oracle
and checks the output class. This is how we know the validator itself can be trusted.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import qa_generate  # noqa: E402
import qa_oracle  # noqa: E402

MODEL = qa_generate.load("qa_model_3d.json")
DATA = qa_generate.load("qa_test_data.json")
INPUTS = {r["input_id"]: r for r in qa_generate.build_inputs(MODEL, DATA)}


def oc(input_id, answer):
    return qa_oracle.evaluate(answer, INPUTS[input_id])["output_class"]


CASES = [
    # TC-QA-01 GetBulk
    ("QA-01-ORIG", "The operation is **GetBulkRequest** (usually called GetBulk). It was introduced in SNMPv2 "
                   "(SNMPv2c) and is not available in SNMPv1. You set non-repeaters and max-repetitions.", "QA-OC-01"),
    ("QA-01-ORIG", "GetBulk came with SNMPv2. SNMPv3 later added authentication and encryption.", "QA-OC-01"),
    ("QA-01-ORIG", "Use GetBulkRequest, which returns many rows in a single request.", "QA-OC-03"),
    ("QA-01-ORIG", "You want the GetBulk operation. It was introduced in SNMPv3 to speed up table walks.", "QA-OC-05"),
    ("QA-01-ORIG", "Walk the table with GetNext, one row at a time.", "QA-OC-07"),
    # TC-QA-02 OID and unit
    ("QA-02-ORIG", "The OID of sysUpTime is 1.3.6.1.2.1.1.3, and you poll the instance 1.3.6.1.2.1.1.3.0. "
                   "Its type is TimeTicks, which counts hundredths of a second since the agent restarted.", "QA-OC-01"),
    ("QA-02-ORIG", "sysUpTime is 1.3.6.1.2.1.1.5.0 and it is measured in hundredths of a second.", "QA-OC-05"),
    ("QA-02-ORIG", "The OID is .1.3.6.1.2.1.1.3.0. The value is measured in milliseconds.", "QA-OC-05"),
    ("QA-02-ORIG", "The OID is 1.3.6.1.2.1.1.3.0.", "QA-OC-03"),
    ("QA-02-PARAM", "hrSystemUptime is 1.3.6.1.2.1.25.1.1.0 in HOST-RESOURCES-MIB. It is TimeTicks, "
                    "so each unit is 1/100 of a second.", "QA-OC-01"),
    ("QA-02-PARAM", "hrSystemUptime is 1.3.6.1.2.1.1.3.0, TimeTicks in hundredths of a second.", "QA-OC-05"),
    # TC-QA-03 counter wrap
    ("QA-03-ORIG", "A 32-bit counter wraps at 2^32 = 4,294,967,296.\nChange = (4,294,967,296 − 4,294,000,000) "
                   "+ 1,000,000 = 1,967,296 octets.\n1,967,296 × 8 = 15,738,368 bits.\n"
                   "15,738,368 / 60 ≈ 262,306 bps (about 262 kbps).", "QA-OC-01"),
    ("QA-03-ORIG", "After the rollover the delta is 1,967,296 bytes, which over 60 s is about 262.3 Kbps.", "QA-OC-01"),
    ("QA-03-ORIG", "The average throughput is roughly 0.262 Mbps.", "QA-OC-01"),
    ("QA-03-ORIG", "Difference = 4,294,000,000 - 1,000,000 = 4,293,000,000 octets, so the rate is "
                   "572,400,000 bps.", "QA-OC-05"),
    ("QA-03-ORIG", "You need to account for the counter wrap at 2^32 before dividing by the interval.", "QA-OC-03"),
    # TC-QA-04 UDP reasons
    ("QA-04-ORIG", "SNMP uses UDP because it is connectionless, so there is no handshake, and it has very low "
                   "overhead. It also keeps working when the network is congested, and the manager handles "
                   "retries and timeouts itself.", "QA-OC-01"),
    ("QA-04-ORIG", "Mostly because UDP is lightweight.", "QA-OC-03"),
    ("QA-04-ORIG", "SNMP normally uses TCP because it needs reliable delivery. UDP is lightweight and "
                   "connectionless.", "QA-OC-05"),
    # TC-QA-05 trap vs inform
    ("QA-05-ORIG", "Both are notifications the agent sends to the manager on UDP port 162. A trap is "
                   "fire-and-forget: the agent gets no acknowledgment. An inform must be acknowledged with a "
                   "Response PDU, and the agent resends it until it is.", "QA-OC-01"),
    ("QA-05-ORIG", "Feature\tTrap\tInform\nAcknowledgment\tNo\tYes\nReliability\tLower\tHigher", "QA-OC-01"),
    ("QA-05-ORIG", "Traps are acknowledged by the manager, while informs are not acknowledged.", "QA-OC-05"),
    ("QA-05-ORIG", "Informs are acknowledged by the receiver.", "QA-OC-03"),
    # TC-QA-06 SNMPv3 levels
    ("QA-06-ORIG", "1. noAuthNoPriv: no authentication and no encryption.\n2. authNoPriv: authentication "
                   "(HMAC-SHA) but no encryption.\n3. authPriv: authentication plus encryption (AES).\n"
                   "For production monitoring a company should use authPriv.", "QA-OC-01"),
    ("QA-06-ORIG", "noAuthNoPriv has no authentication or encryption, authNoPriv adds authentication, and "
                   "authPriv adds authentication and encryption.", "QA-OC-03"),
    ("QA-06-ORIG", "The levels are noAuthNoPriv, authNoPriv and authPriv (authentication and encryption). "
                   "I recommend authNoPriv for production since encryption costs CPU.", "QA-OC-05"),
    # TC-QA-07 command
    ("QA-07-ORIG", "Run:\nsnmpwalk -v2c -c monitor123 10.0.0.1 IF-MIB::ifTable", "QA-OC-01"),
    ("QA-07-ORIG", "Use `snmpwalk -v 2c -c monitor123 10.0.0.1 1.3.6.1.2.1.2.2`.", "QA-OC-01"),
    ("QA-07-ORIG", "snmptable -v2c -c monitor123 10.0.0.1 IF-MIB::ifTable gives it as a table.", "QA-OC-01"),
    ("QA-07-ORIG", "snmpwalk -v2c -c public 10.0.0.1 IF-MIB::ifTable", "QA-OC-05"),
    ("QA-07-ORIG", "snmpwalk -v2c -c monitor123 -q 10.0.0.1 ifTable", "QA-OC-06"),
    ("QA-07-ORIG", "snmpwalk -v2c -c monitor123 10.0.0.1", "QA-OC-03"),
    ("QA-07-ORIG", "Use an SNMP browser and load IF-MIB.", "QA-OC-07"),
    ("QA-07-PARAM", "snmpwalk -v1 -c %s %s IF-MIB::ifTable" % (INPUTS["QA-07-PARAM"]["oracle"]["community"],
                                                             INPUTS["QA-07-PARAM"]["oracle"]["ip"]), "QA-OC-01"),
    ("QA-07-PARAM", "snmpbulkwalk -v1 -c %s %s IF-MIB::ifTable" % (INPUTS["QA-07-PARAM"]["oracle"]["community"],
                                                                 INPUTS["QA-07-PARAM"]["oracle"]["ip"]), "QA-OC-05"),
    # TC-QA-08 nonexistent subject
    ("QA-08-ORIG", "There is no SNMPv4. No RFC defines it. SNMPv3, specified in RFC 3411 to RFC 3418, is "
                   "still the latest version.", "QA-OC-02"),
    ("QA-08-ORIG", "SNMPv4 doesn't exist as a standard. The most recent version is SNMPv3.", "QA-OC-02"),
    ("QA-08-ORIG", "SNMPv4 doesn't exist.", "QA-OC-03"),
    ("QA-08-ORIG", "SNMPv4 is defined in RFC 9561 and adds JSON encoding and TLS transport.", "QA-OC-06"),
    ("QA-08-PARAM", "There is no SNMPv5. SNMPv3 is the current version (RFC 3411-3418).", "QA-OC-02"),
]


@pytest.mark.parametrize("input_id,answer,expected", CASES)
def test_output_class(input_id, answer, expected):
    assert oc(input_id, answer) == expected


def test_param_counter_expected_value():
    t = INPUTS["QA-03-PARAM"]
    _delta, _bits, bps = qa_oracle.expected_rate(t["oracle"])
    assert oc("QA-03-PARAM", "The average is about {:,.0f} bps.".format(bps)) == "QA-OC-01"
    assert oc("QA-03-PARAM", "The average is about 262,306 bps.") == "QA-OC-05"


def test_unfocused_answer_is_partial():
    good = ("The operation is GetBulkRequest. It was introduced in SNMPv2 and is not in SNMPv1. ")
    padded = good + "Some general background on network monitoring. " * 120
    assert oc("QA-01-ORIG", padded) == "QA-OC-04"


def test_empty_answer_fails():
    r = qa_oracle.evaluate("", INPUTS["QA-01-ORIG"])
    assert r["verdict"] == "Fail" and r["errors"] == ["empty answer"]


def test_every_output_class_is_reachable():
    seen = {expected for _i, _a, expected in CASES} | {"QA-OC-04"}
    assert seen == set(MODEL["output"]["classes"])


# Cases taken from the first real run (answers shortened and reworded)
REAL_RUN_CASES = [
    ("QA-05-ORIG", "The main difference is acknowledgment: an SNMP trap is sent without expecting a reply, while "
                   "an SNMP inform requires the receiver to acknowledge it. If an inform is not acknowledged, the "
                   "sender can retry it.", "QA-OC-01"),
    ("QA-05-PARA", "Side-by-side comparison\nAspect SNMP trap SNMP inform\nReceiver response No acknowledgment "
                   "Receiver sends an SNMP Response PDU\nRetransmission No retry Can retry if no response arrives",
     "QA-OC-01"),
    ("QA-05-ORIG", "Aspect SNMP trap SNMP inform\nAcknowledgment None Receiver sends an SNMP Response PDU", "QA-OC-01"),
    ("QA-08-ORIG", "No RFC defines SNMPv4, and SNMPv3 (RFC 3411 to 3418) is still the current version. You may "
                   "see \"version 4\" in some MIB names, such as RFC 6933, which defines Entity MIB (Version 4), "
                   "but that is a MIB module rather than an SNMPv4 protocol standard.", "QA-OC-02"),
    ("QA-08-ORIG", "RFC 9561 defines SNMPv4, which adds JSON encoding.", "QA-OC-06"),
    ("QA-03-ORIG", "The average inbound throughput was 254,629.6 bps (about 254.6 kbps).\nStep by step: "
                   "1,967,296 octets x 8 = 15,738,368 bits, and 15,738,368 / 60 = 262,306.13 bps.", "QA-OC-05"),
    ("QA-03-ORIG", "The average is 262,306 bps, which is about 0.26 Mbps or roughly 0.3 Mbps.", "QA-OC-01"),
    ("QA-03-ORIG", "After the wrap the rate is 262,306 bps on what is probably a 1 Gbps link.", "QA-OC-01"),
    ("QA-03-PARAM", "The increase was\n2\n32\n-4,293,940,403+1,614,298=1,731,191 octets. Over 30 seconds, that is "
                    "about 461,651 bits per second.", "QA-OC-05"),
]


@pytest.mark.parametrize("input_id,answer,expected", REAL_RUN_CASES)
def test_real_run_cases(input_id, answer, expected):
    assert oc(input_id, answer) == expected


def test_site_error_is_not_an_answer():
    import qa_sites
    assert qa_sites.is_site_error("Sorry, something went wrong. Please try your request again.")
    assert not qa_sites.is_site_error("Something went wrong with the counter, so compute the wrap first. " * 5)
