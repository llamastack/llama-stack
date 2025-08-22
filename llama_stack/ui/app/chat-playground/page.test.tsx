import React from "react";
import {
  render,
  screen,
  fireEvent,
  waitFor,
  act,
} from "@testing-library/react";
import "@testing-library/jest-dom";
import ChatPlaygroundPage from "./page";

const mockClient = {
  agents: {
    list: jest.fn(),
    create: jest.fn(),
    retrieve: jest.fn(),
    delete: jest.fn(),
    session: {
      list: jest.fn(),
      create: jest.fn(),
      delete: jest.fn(),
      retrieve: jest.fn(),
    },
    turn: {
      create: jest.fn(),
    },
  },
  models: {
    list: jest.fn(),
  },
  toolgroups: {
    list: jest.fn(),
  },
};

jest.mock("@/hooks/use-auth-client", () => ({
  useAuthClient: jest.fn(() => mockClient),
}));

jest.mock("@/components/chat-playground/chat", () => ({
  Chat: jest.fn(
    ({
      className,
      messages,
      handleSubmit,
      input,
      handleInputChange,
      isGenerating,
      append,
      suggestions,
    }) => (
      <div data-testid="chat-component" className={className}>
        <div data-testid="messages-count">{messages.length}</div>
        <input
          data-testid="chat-input"
          value={input}
          onChange={handleInputChange}
          disabled={isGenerating}
        />
        <button data-testid="submit-button" onClick={handleSubmit}>
          Submit
        </button>
        {suggestions?.map((suggestion: string, index: number) => (
          <button
            key={index}
            data-testid={`suggestion-${index}`}
            onClick={() => append({ role: "user", content: suggestion })}
          >
            {suggestion}
          </button>
        ))}
      </div>
    )
  ),
}));

jest.mock("@/components/chat-playground/conversations", () => ({
  SessionManager: jest.fn(({ selectedAgentId, onNewSession }) => (
    <div data-testid="session-manager">
      {selectedAgentId && (
        <>
          <div data-testid="selected-agent">{selectedAgentId}</div>
          <button data-testid="new-session-button" onClick={onNewSession}>
            New Session
          </button>
        </>
      )}
    </div>
  )),
  SessionUtils: {
    saveCurrentSessionId: jest.fn(),
    loadCurrentSessionId: jest.fn(),
    loadCurrentAgentId: jest.fn(),
    saveCurrentAgentId: jest.fn(),
    clearCurrentSession: jest.fn(),
    saveSessionData: jest.fn(),
    loadSessionData: jest.fn(),
    saveAgentConfig: jest.fn(),
    loadAgentConfig: jest.fn(),
    clearAgentCache: jest.fn(),
    createDefaultSession: jest.fn(() => ({
      id: "test-session-123",
      name: "Default Session",
      messages: [],
      selectedModel: "",
      systemMessage: "You are a helpful assistant.",
      agentId: "test-agent-123",
      createdAt: Date.now(),
      updatedAt: Date.now(),
    })),
  },
}));

const mockAgents = [
  {
    agent_id: "agent_123",
    agent_config: {
      name: "Test Agent",
      instructions: "You are a test assistant.",
    },
  },
  {
    agent_id: "agent_456",
    agent_config: {
      agent_name: "Another Agent",
      instructions: "You are another assistant.",
    },
  },
];

const mockModels = [
  {
    identifier: "test-model-1",
    model_type: "llm",
  },
  {
    identifier: "test-model-2",
    model_type: "llm",
  },
];

const mockToolgroups = [
  {
    identifier: "builtin::rag",
    provider_id: "test-provider",
    type: "tool_group",
    provider_resource_id: "test-resource",
  },
];

describe("ChatPlaygroundPage", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    Element.prototype.scrollIntoView = jest.fn();
    mockClient.agents.list.mockResolvedValue({ data: mockAgents });
    mockClient.models.list.mockResolvedValue(mockModels);
    mockClient.toolgroups.list.mockResolvedValue(mockToolgroups);
    mockClient.agents.session.create.mockResolvedValue({
      session_id: "new-session-123",
    });
    mockClient.agents.session.list.mockResolvedValue({ data: [] });
    mockClient.agents.session.retrieve.mockResolvedValue({
      session_id: "test-session",
      session_name: "Test Session",
      started_at: new Date().toISOString(),
      turns: [],
    }); // No turns by default
    mockClient.agents.retrieve.mockResolvedValue({
      agent_id: "test-agent",
      agent_config: {
        toolgroups: ["builtin::rag"],
        instructions: "Test instructions",
        model: "test-model",
      },
    });
    mockClient.agents.delete.mockResolvedValue(undefined);
  });

  describe("Agent Selector Rendering", () => {
    test("shows agent selector when agents are available", async () => {
      await act(async () => {
        render(<ChatPlaygroundPage />);
      });

      await waitFor(() => {
        expect(screen.getByText("Agent Session:")).toBeInTheDocument();
        expect(screen.getAllByRole("combobox")).toHaveLength(2);
        expect(screen.getByText("+ New Agent")).toBeInTheDocument();
        expect(screen.getByText("Clear Chat")).toBeInTheDocument();
      });
    });

    test("does not show agent selector when no agents are available", async () => {
      mockClient.agents.list.mockResolvedValue({ data: [] });

      await act(async () => {
        render(<ChatPlaygroundPage />);
      });

      await waitFor(() => {
        expect(screen.queryByText("Agent Session:")).not.toBeInTheDocument();
        expect(screen.getAllByRole("combobox")).toHaveLength(1);
        expect(screen.getByText("+ New Agent")).toBeInTheDocument();
        expect(screen.queryByText("Clear Chat")).not.toBeInTheDocument();
      });
    });

    test("does not show agent selector while loading", async () => {
      mockClient.agents.list.mockImplementation(() => new Promise(() => {}));

      await act(async () => {
        render(<ChatPlaygroundPage />);
      });

      expect(screen.queryByText("Agent Session:")).not.toBeInTheDocument();
      expect(screen.getAllByRole("combobox")).toHaveLength(1);
      expect(screen.getByText("+ New Agent")).toBeInTheDocument();
      expect(screen.queryByText("Clear Chat")).not.toBeInTheDocument();
    });

    test("shows agent options in selector", async () => {
      await act(async () => {
        render(<ChatPlaygroundPage />);
      });

      await waitFor(() => {
        const agentCombobox = screen.getAllByRole("combobox").find(element => {
          return (
            element.textContent?.includes("Test Agent") ||
            element.textContent?.includes("Select Agent")
          );
        });
        expect(agentCombobox).toBeDefined();
        fireEvent.click(agentCombobox!);
      });

      await waitFor(() => {
        expect(screen.getAllByText("Test Agent")).toHaveLength(2);
        expect(screen.getByText("Another Agent")).toBeInTheDocument();
      });
    });

    test("displays agent ID when no name is available", async () => {
      const agentWithoutName = {
        agent_id: "agent_789",
        agent_config: {
          instructions: "You are an agent without a name.",
        },
      };

      mockClient.agents.list.mockResolvedValue({ data: [agentWithoutName] });

      await act(async () => {
        render(<ChatPlaygroundPage />);
      });

      await waitFor(() => {
        const agentCombobox = screen.getAllByRole("combobox").find(element => {
          return (
            element.textContent?.includes("Agent agent_78") ||
            element.textContent?.includes("Select Agent")
          );
        });
        expect(agentCombobox).toBeDefined();
        fireEvent.click(agentCombobox!);
      });

      await waitFor(() => {
        expect(screen.getAllByText("Agent agent_78...")).toHaveLength(2);
      });
    });
  });

  describe("Agent Creation Modal", () => {
    test("opens agent creation modal when + New Agent is clicked", async () => {
      await act(async () => {
        render(<ChatPlaygroundPage />);
      });

      const newAgentButton = screen.getByText("+ New Agent");
      fireEvent.click(newAgentButton);

      expect(screen.getByText("Create New Agent")).toBeInTheDocument();
      expect(screen.getByText("Agent Name (optional)")).toBeInTheDocument();
      expect(screen.getAllByText("Model")).toHaveLength(2);
      expect(screen.getByText("System Instructions")).toBeInTheDocument();
      expect(screen.getByText("Tools (optional)")).toBeInTheDocument();
    });

    test("closes modal when Cancel is clicked", async () => {
      await act(async () => {
        render(<ChatPlaygroundPage />);
      });

      const newAgentButton = screen.getByText("+ New Agent");
      fireEvent.click(newAgentButton);

      const cancelButton = screen.getByText("Cancel");
      fireEvent.click(cancelButton);

      expect(screen.queryByText("Create New Agent")).not.toBeInTheDocument();
    });

    test("creates agent when Create Agent is clicked", async () => {
      mockClient.agents.create.mockResolvedValue({ agent_id: "new-agent-123" });
      mockClient.agents.list
        .mockResolvedValueOnce({ data: mockAgents })
        .mockResolvedValueOnce({
          data: [
            ...mockAgents,
            { agent_id: "new-agent-123", agent_config: { name: "New Agent" } },
          ],
        });

      await act(async () => {
        render(<ChatPlaygroundPage />);
      });

      const newAgentButton = screen.getByText("+ New Agent");
      await act(async () => {
        fireEvent.click(newAgentButton);
      });

      await waitFor(() => {
        expect(screen.getByText("Create New Agent")).toBeInTheDocument();
      });

      const nameInput = screen.getByPlaceholderText("My Custom Agent");
      await act(async () => {
        fireEvent.change(nameInput, { target: { value: "Test Agent Name" } });
      });

      const instructionsTextarea = screen.getByDisplayValue(
        "You are a helpful assistant."
      );
      await act(async () => {
        fireEvent.change(instructionsTextarea, {
          target: { value: "Custom instructions" },
        });
      });

      await waitFor(() => {
        const modalModelSelectors = screen
          .getAllByRole("combobox")
          .filter(el => {
            return (
              el.textContent?.includes("Select Model") ||
              el.closest('[class*="modal"]') ||
              el.closest('[class*="card"]')
            );
          });
        expect(modalModelSelectors.length).toBeGreaterThan(0);
      });

      const modalModelSelectors = screen.getAllByRole("combobox").filter(el => {
        return (
          el.textContent?.includes("Select Model") ||
          el.closest('[class*="modal"]') ||
          el.closest('[class*="card"]')
        );
      });

      await act(async () => {
        fireEvent.click(modalModelSelectors[0]);
      });

      await waitFor(() => {
        const modelOptions = screen.getAllByText("test-model-1");
        expect(modelOptions.length).toBeGreaterThan(0);
      });

      const modelOptions = screen.getAllByText("test-model-1");
      const dropdownOption = modelOptions.find(
        option =>
          option.closest('[role="option"]') ||
          option.id?.includes("radix") ||
          option.getAttribute("aria-selected") !== null
      );

      await act(async () => {
        fireEvent.click(
          dropdownOption || modelOptions[modelOptions.length - 1]
        );
      });

      await waitFor(() => {
        const createButton = screen.getByText("Create Agent");
        expect(createButton).not.toBeDisabled();
      });

      const createButton = screen.getByText("Create Agent");
      await act(async () => {
        fireEvent.click(createButton);
      });

      await waitFor(() => {
        expect(mockClient.agents.create).toHaveBeenCalledWith({
          agent_config: {
            model: expect.any(String),
            instructions: "Custom instructions",
            name: "Test Agent Name",
            enable_session_persistence: true,
          },
        });
      });

      await waitFor(() => {
        expect(screen.queryByText("Create New Agent")).not.toBeInTheDocument();
      });
    });
  });

  describe("Agent Selection", () => {
    test("creates default session when agent is selected", async () => {
      await act(async () => {
        render(<ChatPlaygroundPage />);
      });

      await waitFor(() => {
        // first agent should be auto-selected
        expect(mockClient.agents.session.create).toHaveBeenCalledWith(
          "agent_123",
          { session_name: "Default Session" }
        );
      });
    });

    test("switches agent when different agent is selected", async () => {
      await act(async () => {
        render(<ChatPlaygroundPage />);
      });

      await waitFor(() => {
        const agentCombobox = screen.getAllByRole("combobox").find(element => {
          return (
            element.textContent?.includes("Test Agent") ||
            element.textContent?.includes("Select Agent")
          );
        });
        expect(agentCombobox).toBeDefined();
        fireEvent.click(agentCombobox!);
      });

      await waitFor(() => {
        const anotherAgentOption = screen.getByText("Another Agent");
        fireEvent.click(anotherAgentOption);
      });

      expect(mockClient.agents.session.create).toHaveBeenCalledWith(
        "agent_456",
        { session_name: "Default Session" }
      );
    });
  });

  describe("Agent Deletion", () => {
    test("shows delete button when multiple agents exist", async () => {
      await act(async () => {
        render(<ChatPlaygroundPage />);
      });

      await waitFor(() => {
        expect(screen.getByTitle("Delete current agent")).toBeInTheDocument();
      });
    });

    test("hides delete button when only one agent exists", async () => {
      mockClient.agents.list.mockResolvedValue({
        data: [mockAgents[0]],
      });

      await act(async () => {
        render(<ChatPlaygroundPage />);
      });

      await waitFor(() => {
        expect(
          screen.queryByTitle("Delete current agent")
        ).not.toBeInTheDocument();
      });
    });

    test("deletes agent and switches to another when confirmed", async () => {
      global.confirm = jest.fn(() => true);

      await act(async () => {
        render(<ChatPlaygroundPage />);
      });

      await waitFor(() => {
        expect(screen.getByTitle("Delete current agent")).toBeInTheDocument();
      });

      mockClient.agents.delete.mockResolvedValue(undefined);
      mockClient.agents.list.mockResolvedValueOnce({ data: mockAgents });
      mockClient.agents.list.mockResolvedValueOnce({
        data: [mockAgents[1]],
      });

      const deleteButton = screen.getByTitle("Delete current agent");
      await act(async () => {
        deleteButton.click();
      });

      await waitFor(() => {
        expect(mockClient.agents.delete).toHaveBeenCalledWith("agent_123");
        expect(global.confirm).toHaveBeenCalledWith(
          "Are you sure you want to delete this agent? This action cannot be undone and will delete all associated sessions."
        );
      });

      (global.confirm as jest.Mock).mockRestore();
    });

    test("does not delete agent when cancelled", async () => {
      global.confirm = jest.fn(() => false);

      await act(async () => {
        render(<ChatPlaygroundPage />);
      });

      await waitFor(() => {
        expect(screen.getByTitle("Delete current agent")).toBeInTheDocument();
      });

      const deleteButton = screen.getByTitle("Delete current agent");
      await act(async () => {
        deleteButton.click();
      });

      await waitFor(() => {
        expect(global.confirm).toHaveBeenCalled();
        expect(mockClient.agents.delete).not.toHaveBeenCalled();
      });

      (global.confirm as jest.Mock).mockRestore();
    });
  });

  describe("Error Handling", () => {
    test("handles agent loading errors gracefully", async () => {
      mockClient.agents.list.mockRejectedValue(
        new Error("Failed to load agents")
      );
      const consoleSpy = jest
        .spyOn(console, "error")
        .mockImplementation(() => {});

      await act(async () => {
        render(<ChatPlaygroundPage />);
      });

      await waitFor(() => {
        expect(consoleSpy).toHaveBeenCalledWith(
          "Error fetching agents:",
          expect.any(Error)
        );
      });

      expect(screen.getByText("+ New Agent")).toBeInTheDocument();

      consoleSpy.mockRestore();
    });

    test("handles model loading errors gracefully", async () => {
      mockClient.models.list.mockRejectedValue(
        new Error("Failed to load models")
      );
      const consoleSpy = jest
        .spyOn(console, "error")
        .mockImplementation(() => {});

      await act(async () => {
        render(<ChatPlaygroundPage />);
      });

      await waitFor(() => {
        expect(consoleSpy).toHaveBeenCalledWith(
          "Error fetching models:",
          expect.any(Error)
        );
      });

      consoleSpy.mockRestore();
    });
  });
});
